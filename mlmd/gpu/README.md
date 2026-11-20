# omnimd-gpu

The CUDA runtime under the native se_e2_a path: a device handle with a kernel
cache, the neighbor search, and the network kernels the potential and the
trainer are built from. Rust schedules; the arithmetic is CUDA C++ compiled by
NVRTC at run time (and cached on disk), launched through `cudarc`.

| piece | what it does |
|---|---|
| `Device`, `KernelCache` | context, stream, cuBLAS handle; compiled kernels by source hash, in memory and on disk |
| `NeighborSearch`, `NeighborList` | cell-list search with DeePMD's selection, for one system or several that do not see each other; the search keeps its buffers between calls |
| `ScalarNetwork` | embedding nets, value and tangent in one launch |
| `CellTable` | embedding nets as quintic tables in pieces of uneven length, each as long as its net allows (a trained net turns within a hundredth of a unit in some places and not at all in most; the θ of se_e3 spans thousands of units): the piece of an input is found from its sign, its exponent and its first bits, with nothing rounded on the way in float32 or float64. A table may have pieces on one side of zero only, and reach less far than its pieces go; an input beyond is flagged. `CELL_LOOKUP` is the lookup as CUDA source, for kernels that read such tables themselves |
| `GradientNetwork` | fitting nets, forward and backward (few atoms): one launch, or three with a wide first layer in tiles through shared memory and the other layers two rows a block |
| `DenseNetwork` | fitting nets layer by layer on cuBLAS (many atoms) |
| `TrainSet`, `TrainNetwork` | networks with tangents and parameter gradients for the trainer |

## What things cost on the device

Measured on an RTX 4090 (Windows, WDDM, CUDA 12.8) with the probes in
`tests/costs.rs`. They decide how the kernels are laid out, so measure again
on other hardware before carrying the conclusions over: a card with full-rate
double precision, or a driver with cheaper launches, shifts the balance.

```bash
cargo test --release -p omnimd-gpu --test costs -- --ignored --nocapture
```

**Calls from the host** (microseconds each, small buffers):

| call | cost |
|---|---:|
| allocation and free (stream-ordered pool), 8 KB to 256 MiB | 0.4 |
| copy into a device buffer, 8 KB / 128 KB | 5 / 9 |
| clearing a buffer | 5 to 6 |
| kernel launch | 5 to 6 |
| copy back to the host, 8 KB / 128 KB (waits for the stream) | 10 / 20 |
| each further copy back, pageable / pinned | 11 / 4 |
| looking a kernel up in the cache (hashes its source) | about 8 |
| a mark in the stream (`Device::mark`) | 4.5 to 5 |

These are the host's costs, and an evaluation of a small frame does not
wait for the host. One of 88 atoms (se_e2_a with tables, float32) took 166
microseconds before 2026-11-19, read on a probe build with the host's clock
after the calls of every stage and no wait: the host issued its dozen calls
in 40 of them, one after the other, and for 120 it waited in the copy back
while the device worked through what was queued. Two changes that had the
host issue a kernel sooner left the evaluation within 2 microseconds of
where it was (the neighbor search issued before the results are cleared
instead of after; the results copied back to pinned memory). What a queued
call takes the device is in `device_times_by_marks`: a kernel that does
nothing 3.5 microseconds behind another, a mark 1.5. This page said until
then that twenty avoidable calls are a third of a step of a hundred atoms;
that was the host's side added up. The rules stand: one upload and one
download per evaluation (a copy back makes the host wait for all that is
queued, and while the host prepares its first upload the device has nothing
to do), buffers and kernel handles kept between evaluations, status flags
written where the results already go.

So the stages of an evaluation are timed by the device. The host's clock
says when a stage was issued, not how long the device took over it, and a
wait for the device after every stage (how `--profile` worked until
2026-11-19) times another evaluation than the one that runs otherwise. A
mark is an event in the stream that the device stamps when its work gets
there: the time between two marks is the device's over what was issued
between them, with the host waiting for nothing
(`native::gpu_model::StageTimes` says how to read the two clocks together).
A mark is itself a call, though, and eight of them in an evaluation of 150
microseconds double what the host issues: that evaluation took 180
microseconds, and its stages came out up to 5 microseconds longer than
with marks at the two ends of one stage alone. From 200 microseconds on
the two agreed within a microsecond a stage.

**Arithmetic** (nanoseconds per operation on a whole warp):

| operation | one warp alone | multiprocessor busy |
|---|---:|---:|
| float multiply-add, chained / independent sums | 2.0 / 0.65 | about 4 per ns in all |
| double multiply-add, chained / independent sums | 20 / 7 | one per 8 ns in all |
| double division | 450 to 500 | one per 150 ns in all |
| 32-bit integer division by a number the kernel is given | 27 to 29 | one per 2.5 to 3 ns in all |
| double square root | 290 | one per 90 ns in all |
| float to double or back | 15 to 20 | |
| one value from shared memory for the whole warp, with its multiply-add | 1.2 | about 4 per ns in all |
| a value per lane from shared memory, with its multiply-add | | about 1 per ns in all |
| value from global memory, lanes on one line, new line each time | 11 to 22 | 0.36 per ns at 8 warps, 0.65 at 64 |
| the same line again and again (from the first-level cache) | 8 to 12 | 0.65 to 0.8 per ns |
| value from global memory, every lane on a line of its own | 24 to 110 | |

Double precision runs at about 1/64 of the single precision rate on this
card, and a division costs sixty multiply-adds. Loads from global memory are
slow even when the data is cached. Shared memory is five to ten times faster
for a value the whole warp shares and about twice as fast for a value per
lane.

What follows for the kernels:

- Ask the device for nothing in the middle of an evaluation. A copy back to
  the host waits for all that is queued before it, and the device then
  stands still while the host queues the rest. An se_e3 model brought the
  neighbor counts of its atoms back to lay out the rows of their pairs: 17
  to 30 µs an evaluation, an eighth of a frame of 88 atoms in float32 with
  tables (154 µs). Where the host needs such a number only for the size of
  buffers, it can take the most the number can be (an allocation costs
  the same at any size) and leave the number on the device: a kernel of
  one block lays the rows of up to 1,024 atoms out with a scan over its
  shared memory. A block has one multiprocessor, though, so keep what
  each of its threads loads small (the neighbors of a type fill its first
  slots: their number is found by halving, not by counting).
- Divide once. Reciprocals of constants come from the host (`1 / dstd`,
  `1 / (rcut - rcut_smth)`) or are formed once per thread (`1 / slots`,
  `1 / r` per neighbor); neighbor distances are taken from the search, not
  from another square root. Integers too: a division by the cells of an
  axis takes a warp as long as fourteen float multiply-adds one after the
  other, and where one warp does a center's work from end to end the center
  waits for each. The neighbor search had the cell and the periodic image
  of every cell of a center's stencil from a division for each axis, 81 for
  a stencil of 27 cells. It divides once for an axis now and steps from
  there, and it takes the cells along the last axis, which lie one after
  the other in the grid with their atoms, as one run of atoms. A search of
  88 atoms with its packing and a wait for the device went from 37 to 30
  microseconds, one of 5,632 atoms from 313 to 280
  (`neighbor_search_costs`).
- Try a block for an atom with more warps where the blocks are few. The
  force kernel of the two-body descriptors takes the neighbor slots of its
  atom a warp each, the warps of the block in turns: eight warps are
  through 212 slots in 27 turns where four take 53. With blocks of 256
  threads instead of 128 a float32 evaluation is 4% to 6% shorter at 54 to
  108 blocks and 1.2% to 5% at 162 to 352, and 1.6% to 3.2% longer at 528,
  896 and 12,288. In float64 an evaluation with se_e2_r is 1.6% to 4.4%
  shorter up to 224 blocks and within 1% at 264 to 352, and one with
  se_e2_a stays within 1.7% at every size but one. So the width goes by the
  centers of the pass and the precision
  (`native::gpu_model::WIDE_FORCE_CENTERS_F32` and
  `WIDE_FORCE_CENTERS_F64`). While every warp of a block reduced its own
  sums in double precision (see below), a block of eight warps did twice
  the reductions of one of four, and the gain ended at 224 blocks in
  float32 and at 108 in float64. Blocks of 512 threads were no better than
  256 at 88 atoms and worse at 224, and wider blocks for the descriptor
  kernel changed nothing. Why the gain ends where it does was not measured.
- Give a chain of one center's work to several warps. The neighbor search
  had a warp for a center, which walked the columns of the center's stencil
  one after the other (27 cells in 9 columns for the 88-atom frame).
  Stopped after each of its steps on a probe build, the kernel for 88 atoms
  took 5.3 µs to launch and return at once (back to back, the host's pace),
  14.8 through the scan and 19.7 in all. It has a block of four warps for a
  center now: the warps take the columns in turns and append to one buffer
  of candidates through a count in shared memory, and the double precision
  pass and the ranking take the candidates 128 at a time. A slot follows
  from the ranks alone, so the order in which the warps fill the buffer does
  not matter and the list is the same, bit for bit, with any count of warps.
  The kernel for 88 atoms went to 12.8 µs (8.0 through the scan), the one
  for 5,632 from 114 to 108, where the blocks of one warp already filled the
  device. Evaluations of the HAP frames of 88 to 704 atoms in float32 with
  tables are 2 to 7 µs shorter (two runs: 155.0 → 149.8 µs and 152.5 →
  150.3 for 88 atoms, less than the kernel's 7 µs), and an MD step of 88
  atoms 2.4 to 4.7 µs shorter; from 1,536 atoms on the difference is within
  the scatter of the runs. Two warps were up to 7 µs an evaluation slower
  than four below a thousand atoms, and eight were within the scatter of
  four up to 224 atoms and slower from 704 on, the kernel a third longer for
  5,632 atoms (`WARPS_PER_CENTER` in `src/nlist.rs`).
- Price a list kept across steps before keeping one. With a Verlet skin a
  search would take its candidates to rcut + skin, and until the two atoms
  that have moved most since have moved more than the skin between them,
  each step would measure those candidates again instead of scanning. In
  NVT runs of the three HAP frames with the trained model (0.5 fs steps,
  2,000 of them, at 300 and 500 K) such a list lasted 9 to 24 steps with a
  skin of 0.5 Å, 21 to 125 with 1 Å and 59 to 2,000 with 2 Å. A step would
  save less than the scan, though, and with four warps a center the scan is
  short. Read back instead of scanned, the 71 to 78 candidates within the
  cutoff itself take about 3 to 4.5 µs less than a fresh search (12.3 to
  13.2 µs), and the 112 to 124 candidates of a 1 Å skin take 11.9 to 14.3
  µs, within the scatter of the probe of a fresh search; that against
  evaluations of 150 to 210 µs. A skin would also spare most of the host's
  packing of the search input, 2 to 6 µs at these sizes. So for the HAP
  frames the search runs afresh every step. Thousands of atoms are another
  case: for 5,632 the packing takes 125 to 167 µs, more than the kernel's
  108, and a 1 Å skin reads back in 100, so a skin or a faster packing
  could save about 5% of an evaluation of 2,770 µs. The packing is the part
  of the search to look at next.
- Keep double precision for what needs it. The neighbor search scans in
  single precision with a margin that covers its rounding and measures only
  the surviving candidates in double; distances are ranked by their bit
  patterns as integers. A float32 model keeps double precision for its
  geometry only, the environment rows as they are worked out and their
  Jacobian for the forces: the rows as they are stored, the sums that make B,
  the descriptor and the adjoint of B are single. Those sums are most of
  what the descriptor kernel does, four multiply-adds for every neighbor and
  embedding output: for 98,304 water atoms it took 11.9 ms with them in
  double and 5.6 ms in single.
- In double precision, give every lane of a warp work. The table above
  counts a multiply-add for a whole warp, and that is what it costs: an
  instruction in double precision takes its eight nanoseconds of the
  multiprocessor with one lane at work or with thirty-two. The kernels that
  read the tables of se_e3 had a warp for a row at first, the lanes over
  the forty outputs of the row, which is two rounds of instructions with
  three in four lanes idle in the second, and what a row needs once was
  worked out by all thirty-two. With a thread for a row, alone over the
  outputs, the same work took a third of the time and less (192 water atoms
  with 3,600 rows each in float64: the components 0.42 against 1.2 ms, the
  force stage 0.49 against 2.6). Handing the per-row work from one lane to
  the others changed nothing (4.17 against 4.22 ms an evaluation): idle
  lanes cost as much as busy ones. In single precision the same kernels are
  bound by their loads, and a warp that reads the six numbers of every
  output of a row together is as fast or faster.
  The force kernels had two such places even in float32. In those of the
  two-body descriptors a warp summed what a neighbor slot gives over the
  outputs of the embedding, and its first lane turned the sums into double
  and scaled them while the other warps of the block waited. And in all of
  them every warp reduced the twelve sums of its block (the force on its
  atom and the virial) over its lanes, five additions each with half the
  lanes at work, then a quarter: sixty a warp. The sums of a slot now go to
  shared memory as they are and the thread that has the slot in the next
  step turns them into double, and the threads put down what they hold for
  the block in shared memory, where a warp for each of the twelve adds up
  the warps lane by lane and reduces once. In place in an evaluation in
  float32 the force kernel of se_e2_a went from 23 to 19 µs for 88 atoms
  and from 33 to 25 for 224. The room for the block's sums, twelve doubles
  a thread, is not free: with se_e2_a an evaluation of 528 centers was 2.8%
  to 3.7% longer with it, so such passes go without
  (`native::gpu_model::BLOCK_SUMS_CENTERS`).
- Reduce over a block only after a barrier of the block. Where an atom has
  more neighbors than its block has threads, the threads leave the loop
  over the slots at different times. The force kernel of se_e3 (blocks of
  256 threads) had the warps reduce their sums as they came out of it, and
  the force on an atom with 257 to 300 neighbors came out 4e-9 to 1.5e-8
  eV/Å off, in every run alike; a barrier of the block before the
  reductions set it right. Why was not found: a barrier of the warp in that
  place set it right for one model and not for another (`block_sums` in
  `native/src/gpu_model.rs` says what else was tried).
- Give a launch the shared memory its kernel lays out, exactly, for nothing
  reports a small shortfall. On this card a block wrote up to 1,088 bytes
  past the size it was launched with and no test failed, and
  compute-sanitizer's memcheck reported only the accesses from 1,024 bytes
  past it on; a block that wrote up to 24 KB past it stopped the device on
  an illegal address. Mutants that left a region out of the size of a force
  kernel's shared memory (96 bytes, up to 320, and 1,088) passed every test,
  and memcheck found only the last. The size functions beside the kernels
  (`angular_force_shared`, `radial_force_shared` and the others in
  `native/src/gpu_model.rs`) are the one place each layout is counted, for
  the potential and the trainer alike.
- Past the arithmetic stands the memory. With many atoms the stages that
  write the embedding rows or read them (100 values a neighbor) move at its
  rate: the descriptor reads 3.5 GB in 5.6 ms for 98,304 atoms, and nothing
  done to its other loads changed that.
  With few atoms the latency of a load counts instead: copying an atom's
  environment rows and neighbor counts to shared memory before the sums took
  5% off a pass of 88 atoms and nothing off one of 98,304.
- Do no work for empty neighbor slots. With `sel` sized for the densest
  environment most slots are empty (65% in the HAP systems); their embedding
  row is the same for a pair of types, so it is evaluated once per model and
  enters the descriptor as one term.
- Check what the compiler makes of a branch. A branch whose body is a few
  double precision operations gets turned into straight-line code that runs
  for every element and discards the result. The tie-breaking of the
  neighbor ranking cost forty times the ranking itself that way; it now sits
  in a `__noinline__` function behind a count of ties. A branch inside a
  loop that chooses between a cheap and a dear way to the same value did no
  better: the dear one kept running. Such a choice belongs outside the loop,
  as two pieces of code.
- A dense layer that reads its weights from global memory is bound by those
  loads, not by arithmetic: the fitting kernel spent about 20 ns per weight
  line and warp, a row at a time. So the wide first layer of the fitting
  nets (three quarters of their arithmetic) goes through shared memory in
  tiles when the rows are grouped by network
  (`GradientNetwork::forward_gradient_grouped`): a tile of the weights is
  copied once and serves every row of a chunk, a warp does two rows and a
  lane eight outputs, so one read of a weight feeds two products and one
  read of an input sixteen. The first layer's transpose for the gradient is
  tiled the same way, a warp doing two rows there too. The other layers take
  two rows a block, so that a weight read from global memory serves both
  (the next point). Measured for 1600 -> 240 -> 240 -> 240 -> 1 nets (probe
  `fitting_costs`): 1.7 to 2.4 times faster than the one kernel at 88 to
  2,816 rows in float32. In double precision the tiles are slower (the
  arithmetic is the bound, and the extra launches cost), so they are off
  there.
- Sharing a load between rows pays as far as the multiprocessors keep
  enough warps at work. For the layers after the first, two rows a block
  are faster than one at every number of rows (88 rows: 31 against 35
  microseconds; 704: 67 against 136; 2,816: 407 against 576), though a block
  of two rows takes about as long as one of one. Four and eight rows a block
  are no faster than two (2,816 rows: 421 and 479): a block takes four or
  eight times as long, and its shared memory, which grows with the rows,
  leaves room for fewer blocks on a multiprocessor. The same for the tile of
  the first layer's transpose: with two rows a warp it went from 0.19 to
  0.15 microseconds a row with many rows. With both, the fitting stage went
  from 341 to 255 microseconds for 704 rows, from 655 to 496 for 1,408 and
  from 75 to 67 for 88. What is left with many rows is 0.07 microseconds a
  row in the forward tiles, 0.14 in the layers after the first and 0.15 in
  the backward tiles.
- What was tried for those tiles and did not pay: four rows a warp in the
  transposed pass (fewer reads per product, but too few warps left to keep a
  multiprocessor busy), and eight inputs a lane with the outputs reloaded in
  turn (no gain at 88 rows, a third faster at 224 and more: worth taking up
  if systems of that size matter).

## Batches of frames, committees of models

A system of a hundred atoms leaves most of the card idle: its evaluation is a
dozen launches that each wait for the one before, and none of them fills the
multiprocessors. More work per launch costs little, so the potential built on
this crate (`omnimd_native::gpu_model::GpuPotential`) takes two kinds of it:

- **Several frames in one pass.** The atoms of all frames form one list,
  sorted by type. Only the neighbor search knows that they belong to
  different systems (`NeighborSearch::run_systems`): every system has its
  cell, its grid and its scan margin in a header of the search input, and a
  center scans the cells of its own system. Each atom gets, bit for bit, the
  neighbors it gets when its frame is searched alone. Every later kernel
  works center by center and does not ask which frame a center is from; the
  host adds up atomic energies and virials per frame after the one download.
- **Several models in one pass.** Models of one architecture, such as an
  ensemble kept for model deviation, are the second dimension of the launch
  grid. Model `m` reads the same neighbor lists, finds its statistics and
  networks at center type `m * ntypes + type` (the network sets simply hold
  all models' nets), and has its work rows and results after those of the
  models before it. The neighbors are searched once for all models.

The two combine: a batch of frames for a committee is still one pass.

What does not change with the batch: launches, uploads and the download (one
each per stage), the host's binning of atoms into cells (per system, a few
microseconds each), and the rule that picks the fitting path, which goes by
the atoms per model in a chunk. What a batch cannot share is the arithmetic,
so the gain ends where the card is full; see the
[timings](../bench/results/SUMMARY.md).

A frame of a batch that leaves the range of the embedding tables does not
drag the others to the exact networks: the batch is then evaluated a frame at
a time, each with its own fallback.

## Tests

`tests/device.rs` compares every kernel with a plain CPU reference: products
against loops, networks against chains with finite-difference tangents and
gradients, the neighbor search against brute force over images (random and
skewed cells, boxes narrower than the cutoff, open systems, truncation, exact
ties on a lattice, pairs a part in 1e9 from the cutoff at coordinates where
single precision cannot tell them apart), a search over several systems
against each system searched alone, and searches with 2 to 16 warps per
center against one warp, bit for bit and with the status of every center
(also for forty atom types, more than a warp has lanes). The tests skip
without a CUDA driver; `OMNIMD_REQUIRE_GPU=1` turns a missing device into a
failure.
