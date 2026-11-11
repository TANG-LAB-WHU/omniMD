# omnimd-gpu

The CUDA runtime under the native se_e2_a path: a device handle with a kernel
cache, the neighbor search, and the network kernels the potential and the
trainer are built from. Rust schedules; the arithmetic is CUDA C++ compiled by
NVRTC at run time (and cached on disk), launched through `cudarc`.

| piece | what it does |
|---|---|
| `Device`, `KernelCache` | context, stream, cuBLAS handle; compiled kernels by source hash, in memory and on disk |
| `NeighborSearch`, `NeighborList` | cell-list search with DeePMD's selection; the search keeps its buffers between calls |
| `ScalarNetwork`, `ScalarTable` | embedding nets (value and tangent in one launch), exactly or as quintic tables |
| `GradientNetwork` | fitting nets, forward and backward (few atoms): one launch, or three with a wide first layer in tiles through shared memory |
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
| allocation and free (stream-ordered pool) | 0.4 |
| copy into a device buffer, 8 KB / 128 KB | 5 / 9 |
| clearing a buffer | 5 to 6 |
| kernel launch | 5 to 6 |
| copy back to the host, 8 KB / 128 KB (waits for the stream) | 10 / 20 |
| each further copy back, pageable / pinned | 11 / 4 |
| looking a kernel up in the cache (hashes its source) | about 8 |

A molecular dynamics step of a hundred atoms takes a few hundred microseconds,
so twenty avoidable calls are a third of it. Hence: one upload and one
download per evaluation, buffers and kernel handles kept between evaluations,
status flags written where the results already go.

**Arithmetic** (nanoseconds per operation on a whole warp):

| operation | one warp alone | multiprocessor busy |
|---|---:|---:|
| float multiply-add, chained / independent sums | 2.0 / 0.65 | about 4 per ns in all |
| double multiply-add, chained / independent sums | 20 / 7 | one per 8 ns in all |
| double division | 450 to 500 | one per 150 ns in all |
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

- Divide once. Reciprocals of constants come from the host (`1 / dstd`,
  `1 / (rcut - rcut_smth)`) or are formed once per thread (`1 / slots`,
  `1 / r` per neighbor); neighbor distances are taken from the search, not
  from another square root.
- Keep double precision for what needs it. The neighbor search scans in
  single precision with a margin that covers its rounding and measures only
  the surviving candidates in double; distances are ranked by their bit
  patterns as integers.
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
  read of an input sixteen. The other layers stay row by row; the first
  layer's transpose for the gradient is tiled the same way. Measured for
  1600 -> 240 -> 240 -> 240 -> 1 nets: 1.5 to 1.8 times faster than the one
  kernel at 88 to 1024 rows in float32. In double precision the tiles are
  slower (the arithmetic is the bound, and the extra launches cost), so they
  are off there.
- What was tried for those tiles and did not pay: four rows a warp in the
  transposed pass (fewer reads per product, but too few warps left to keep a
  multiprocessor busy), and eight inputs a lane with the outputs reloaded in
  turn (no gain at 88 rows, a third faster at 224 and more: worth taking up
  if systems of that size matter).

## Tests

`tests/device.rs` compares every kernel with a plain CPU reference: products
against loops, networks against chains with finite-difference tangents and
gradients, the neighbor search against brute force over images (random and
skewed cells, boxes narrower than the cutoff, open systems, truncation, exact
ties on a lattice, pairs a part in 1e9 from the cutoff at coordinates where
single precision cannot tell them apart). The tests skip without a CUDA
driver; `OMNIMD_REQUIRE_GPU=1` turns a missing device into a failure.
