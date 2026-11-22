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
| `CellTable` | embedding nets as quintic tables in pieces of uneven length, each as long as its net allows (a trained net turns within a hundredth of a unit in some places and not at all in most; the θ of se_e3 spans thousands of units): the piece of an input is found from its sign, its exponent and its first bits, with nothing rounded on the way in float32 or float64. A table may have pieces on one side of zero only, and reach less far than its pieces go; an input beyond is flagged. `CELL_LOOKUP` is the lookup as CUDA source, for kernels that read such tables themselves, with `cell_value` and `cell_tangent` (the quintic of a piece and its slope), which the kernel of the tables evaluates too, so that a kernel that looks a value up gets the number that kernel would have stored |
| `GradientNetwork` | fitting nets, forward and backward (few atoms): one launch, or three with a wide first layer in tiles through shared memory and the other layers two rows a block |
| `GroupedNetwork` | fitting nets layer by layer on cuBLAS (many atoms), all nets of a layer in one grouped product (cuBLAS 12.5 and later) |
| `DenseNetwork` | fitting nets layer by layer on cuBLAS a net at a time (many atoms, older cuBLAS) |
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
| the same behind a running kernel, 3 KB / 512 KB (from pageable memory, see below) | 3.5 / 15 |
| clearing a buffer | 5 to 6 |
| kernel launch | 5 to 6 |
| copy back to the host, 8 KB / 128 KB (waits for the stream) | 10 / 20 |
| each further copy back, pageable / pinned | 11 / 4 |
| looking a kernel up in the cache (hashes its source) | about 8 |
| a mark in the stream (`Device::mark`) | 4.5 to 5 |

A copy from pageable memory queued behind a running kernel does not make
the host wait for the kernel: 64 copies of 3 or of 40 KB in a row each
returned within 9 microseconds behind a kernel of 4 ms, while of copies of
512 KB the third waited for it, and so did a single one of 8 MB
(`upload_behind_a_busy_stream`, 2026-11-21).

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
  threads instead of 128 a float32 evaluation with the tables stored was 4%
  to 6% shorter at 54 to 108 blocks and 1.2% to 5% at 162 to 352, and 1.6%
  to 3.2% longer at 528, 896 and 12,288 (2026-11-19). Since the kernels of
  se_e2_a look the tables up in float32 (2026-11-21), the device keeps 256
  of the HAP model's wide blocks with the sums at once instead of 384, and
  from 264 blocks they lose there (next item); of the water model's it
  still keeps 384. In float64 an evaluation with se_e2_r is 1.6% to 4.4%
  shorter up to 224 blocks and within 1% at 264 to 352, and one with
  se_e2_a stays within 1.7% at every size but one. While every warp of a
  block reduced its own sums in double precision (see below), a block of
  eight warps did twice the reductions of one of four, and the gain ended
  at 224 blocks in float32 and at 108 in float64. Blocks of 512 threads
  were no better than 256 at 88 atoms and worse at 224, and wider blocks
  for the descriptor kernel changed nothing.
- Count the blocks the device keeps at once before choosing a block. A
  multiprocessor keeps as many blocks of a kernel as its shared memory,
  registers and threads hold, and a pass with more than the multiprocessors
  (128) times that waits for a second wave, however short each block is. In
  float32 the wide blocks with the sums of a block across its warps (below)
  gain while the device keeps every block of the pass at once, and lose once
  it does not: with the HAP model and tables, 2.2% to 4.6% less than the
  narrow ones with the sums up to 224 blocks, 1.1% to 5.0% more from 264 to
  507 (it keeps 256 of them). Without the sums the wide blocks stop gaining
  sooner, by 264 blocks, where the device still keeps them all (384); in
  float64 the two widths with the sums come out within 1.5% of each other
  while both fit. The driver says how many a multiprocessor keeps of a
  kernel for a size of block and its shared memory
  (`cuOccupancyMaxActiveBlocksPerMultiprocessor`): of the force kernel of
  the HAP se_e2_a model in float32 with tables, two wide blocks with the
  sums, four narrow ones with them, three wide and seven narrow ones
  without; of the water model three wide ones with the sums. So the force
  kernels of se_e2_a and se_e2_r take the first of these four ways that
  keeps all the blocks of the pass at once (`native::gpu_model::force_way`),
  and past every way the narrow blocks, with the sums for se_e2_r only.
  Against the limits of centers that chose before (`WIDE_FORCE_CENTERS_F32`,
  `_F64` and `BLOCK_SUMS_CENTERS`, kept for a driver that does not say), the
  choice changed in 31 of 64 cases: in float32 an evaluation was 0.4% to
  4.1% shorter in all 23 (1.4% to 4.1% in the ten from 338 to 384 centers, a
  committee of four on 88 atoms among them), and in float64 the 8 came out
  between 1.3% shorter and 0.4% longer (two runs, against the lesser of two
  runs of the build before). One number of centers could not serve both
  models: HAP keeps two wide blocks with the sums at once where water keeps
  three.
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
  atoms 2.4 to 4.7 µs shorter. From 1,536 atoms on the differences are a
  fraction of a percent. For 12,288 water atoms in float64 four warps came out
  slowest by the least in three runs, by 26 to 164 µs (0.04% to 0.27%),
  while the runs of the build before differed by 41 and 138, so the size of
  that is not established. Two warps were up to 7 µs an evaluation slower
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
  could save about 5% of an evaluation of 2,770 µs. So the packing came
  next.
- Pack the search input in one pass, into memory the search keeps. The
  host binned the atoms of every system into a grid built anew for each
  search: it allocated a copy of the positions, their fractions, wrapped
  positions and cells, and the four-byte entries of the input, which it
  then paired into words, and it took each floor of a fractional
  coordinate, three an atom, from the C library (the x86-64 baseline has no
  rounding instruction). It now places each atom once and writes the
  entries straight into the words it uploads, through a view of them as
  32-bit halves; the order of the atoms, the grids and the counts of the
  counting sort stay with the search between calls (`Scratch` in
  `src/nlist.rs`), and on a processor with SSE4.1 a copy compiled with it
  takes the floors in one instruction. The input is the same, bit for bit:
  a test holds it to the packing it replaced (`src/nlist/reference.rs`) on
  more than 400 inputs. On a performance core of the i9-14900K the packing
  takes 43% to 48% less time from 88 to 12,288 atoms (1.8 → 1.0 µs for 88
  atoms, 4.4 → 2.4 for 224, 115 → 61 for 5,632, 232 → 124 for 12,288
  water atoms), and a search with its packing and a wait for the device
  (`neighbor_search_costs`) went from 25.3 to 20.5 µs for 88 atoms, 49.0 to
  41.2 for 704 and 282 to 237 for 5,632. An evaluation in float32 with
  tables is 4.2 and 5.5 µs shorter for the HAP frames of 169 and 224 atoms
  (2.3% and 2.6%), 9.7 for 704 and 47.5 for 5,632 (1.9% and 1.7%). An MD
  step of 169 atoms went from 206.4 to 200.8 µs, the build before reading
  207.9 the second time, and one of 224 atoms from 219.9 to 214.1, against
  216.6 the second time. The evaluations of 169 and 224 atoms gain more
  than the packing saves at those sizes when it is timed in a loop of its
  own (1.6 and 2.0 µs); why was not measured. For
  88 atoms and for the water frames in float64 the differences are about
  those of the build before from itself. Issued back to back without a
  wait, the search of 704 atoms takes 42.5 µs where it took 21.2: the host
  now reaches the next upload while the kernel still runs, and the times
  fit an upload of 40 KB from pageable memory that then waits for the
  stream (the driver API's notes on synchronization allow an asynchronous
  copy that stages pageable memory to). A later probe did not reproduce
  such a wait (the table of host calls above), so what the searches wait
  for there is open. Evaluations, MD steps and the
  trainer wait for their results before the next search, so they never
  issue searches that way. An upload from a pinned buffer of the search
  made evaluations of the HAP frames 4 to 7 µs slower and is not used
  (`bench/results/SUMMARY.md`).
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
  a thread, is not free: it leaves a multiprocessor room for fewer blocks,
  and with se_e2_a an evaluation of 528 centers was 2.8% to 3.7% longer
  with it. So the two-body kernels form the sums by the blocks the device
  keeps at once, together with the width of the blocks (above).
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
- In float32, evaluate a table where its value is used. With tables, a
  kernel evaluated the quintics of se_e2_a's embedding for every neighbor
  slot and stored each value and its slope, 200 numbers a slot with 100
  outputs; the descriptor kernel read the values back and the force kernel
  both. Over a million rows of 100 outputs (`embedding_table_costs`,
  nanoseconds a row): in float32, storing both took 1.04 to 1.10 and
  reading them back 0.49 for the values and 0.95 for both, about 2.5 in
  all, where evaluating the quintics in the kernels that use them took 0.34
  to 0.52 for the values and 0.36 to 0.52 for values and slopes, 0.7 to 1.0
  in all. In float64 the arithmetic outweighs the memory: 2.5 for the values
  and 5.8 for both against 3.7 to store and 1.0 and 1.95 to read back, 8.3
  against 6.6. So in float32 the descriptor and force kernels of se_e2_a
  find the piece and the place of each slot once and evaluate the quintics
  of its outputs themselves, and in float64 they read stored values
  (`native::gpu_model::GpuPotential::inline_tables`). Evaluations in
  float32 with tables became 5% to 8% shorter for the HAP frames of 88 to
  224 atoms, 37% for 5,632 atoms and 40% for 12,288 water atoms. In float64,
  looking the tables up made evaluations 2.3% and 1.7% longer for 88 HAP
  atoms and 192 water atoms and 1.3% shorter for 5,632 HAP atoms.
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
- Past some hundreds of rows cuBLAS does the fitting nets faster than the
  fused kernels, but a net at a time it costs a fixed amount per net:
  about nineteen calls for each center type and model, some 0.6 ms for the
  five types of HAP whatever the rows (a probe, not committed: 767
  microseconds for 5,632 rows split over four nets, 382 for the same rows
  in one). Streams of their own for the nets took that only to 615.
  Grouped products (`cublas<t>gemmGroupedBatched`) do a layer of all nets
  in one call, so `GroupedNetwork` issues six products and five small
  kernels for three hidden layers however many nets there are; the
  activations, the output layer and the residual sums of the gradient are
  in the kernels, and the device's arrays of matrix addresses are uploaded
  again only when the layout of a chunk changes (a few KB from pageable
  memory, which the host issues without waiting for the kernels queued
  before it: `upload_behind_a_busy_stream` in `tests/costs.rs`). A grouped
  call takes the device 20 to 24 microseconds whatever its size, which
  puts the path some 0.2 ms behind the fused kernels at one row; they meet
  near 700 to 900 rows. Whole passes in float32 with tables (fused,
  grouped, a net at a time, in microseconds): HAP frames of 169 atoms,
  2,704 rows 1,831, 1,278 and 1,660; four models on them, 676 rows 490,
  529 and 2,947 and 10,816 rows 9,253, 3,699 and 5,979. With two nets
  (water) a net at a time is 20 to 30 microseconds faster up to some 1,900
  rows and the grouped products up to 90 faster beyond, so the potential
  takes them from three nets on and, with fewer, from 1,920 rows
  (`GROUPED_FITTING_MIN_NETS` in `native/src/gpu_model.rs`).
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
microseconds each). The rule that picks the fitting path goes by the rows
of the fitting nets in a chunk, its atoms times the models, so a batch or a
committee leaves the fused kernels for the grouped products as soon as the
products are faster (`FUSED_FITTING_ROWS_F32` in `native/src/gpu_model.rs`).
With fewer than three nets in a chunk the fused kernels keep their limit of
before, 340 atoms a center type, and the grouped products take over from a
net at a time at 1,920 rows. What a batch cannot share is the arithmetic, so
the gain ends where the card is full; see the
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
