# The neighbor search with four warps a center, and a skin priced, 2026-11-20

The request was to keep the neighbor list across MD steps with a Verlet
skin. Before building one, this section priced it: how many steps a list
with a skin lasts in MD of the HAP frames, and what a step that measures the
stored candidates again, instead of scanning the stencil, would save. The
scan was most of the search kernel, because one warp walked the whole
stencil of its center. With a block of four warps a center the scan is
short, and for the HAP frames a skin would then save at most about 3 to 4.5
µs of an evaluation of 150 to 210 µs, and about nothing with a skin wide
enough to last. The four warps are in; the skin is not built for these
sizes. For thousands of atoms the host's packing of the search input is
the larger part (below).

**How long a list with a skin lasts.** NVT molecular dynamics (CSVR, τ 50
fs, 0.5 fs steps, 2,000 steps) of the three HAP frames of the engine
benchmarks with the trained se_e2_a model in float32 with tables, at 300 and
500 K, by the engine of 475cafb. From each build of the list, the steps
until the two atoms that have moved most since then have together moved
more than the skin (the cell is fixed); steps per build, the first build
counted (`hap-skin-intervals-rtx4090-2026-11-20.json`):

| frame | K | largest step of an atom (Å) | skin 0.5 Å | skin 1 Å | skin 2 Å |
|---|---:|---:|---:|---:|---:|
| 88 atoms | 300 | 0.029 | 23.5 | 125.0 | 2000 |
| 88 atoms | 500 | 0.040 | 17.7 | 58.8 | 2000 |
| 169 atoms | 300 | 0.037 | 11.9 | 27.4 | 111 |
| 169 atoms | 500 | 0.052 | 9.6 | 21.3 | 59 |
| 224 atoms | 300 | 0.035 | 12.0 | 69.0 | 2000 |
| 224 atoms | 500 | 0.046 | 9.4 | 27.8 | 2000 |

**Where the time of the search kernel goes.** A probe build stops the
kernel after each of its steps; marks around 3,000 launches back to back
(500 for 5,632 atoms) after 20 to warm up, so a launch that returns at once
takes the host's pace. Cutoff 6 Å, sel [44, 16, 60, 72, 20]; 704 and 5,632
atoms are the 88-atom frame replicated 2 and 4 times along each axis (µs a
launch; the stops scatter by a microsecond or two, so a later stop can read
less than an earlier one; `hap-search-stages-rtx4090-2026-11-20.json`):

| atoms | warps | launch and return | through the scan | through exact distances | all | candidates a center |
|---:|---:|---:|---:|---:|---:|---:|
| 88 | 1 | 5.3 | 14.8 | 17.2 | 19.7 | 74 |
| 88 | 2 | 4.8 | 11.2 | 13.1 | 14.5 | 74 |
| 88 | 4 | 4.7 | 8.0 | 9.4 | 12.8 | 74 |
| 88 | 8 | 8.7 | 9.2 | 8.6 | 12.2 | 74 |
| 169 | 1 | 5.4 | 13.6 | 14.9 | 18.6 | 71 |
| 169 | 2 | 7.1 | 10.7 | 12.7 | 14.3 | 71 |
| 169 | 4 | 6.5 | 10.1 | 9.1 | 12.3 | 71 |
| 169 | 8 | 5.4 | 8.4 | 10.9 | 12.7 | 71 |
| 224 | 1 | 5.6 | 15.8 | 14.7 | 19.1 | 78 |
| 224 | 2 | 6.9 | 10.2 | 12.2 | 14.7 | 78 |
| 224 | 4 | 6.7 | 9.4 | 9.1 | 13.2 | 78 |
| 224 | 8 | 6.4 | 8.3 | 8.8 | 13.2 | 78 |
| 704 | 1 | 6.9 | 15.2 | 19.7 | 25.8 | 74 |
| 704 | 2 | 5.0 | 12.5 | 15.5 | 21.7 | 74 |
| 704 | 4 | 5.0 | 12.6 | 14.3 | 19.4 | 74 |
| 704 | 8 | 6.3 | 13.1 | 18.7 | 25.7 | 74 |
| 5,632 | 1 | 7.0 | 50.8 | 73.8 | 114.5 | 74 |
| 5,632 | 2 | 7.4 | 35.8 | 61.9 | 104.5 | 74 |
| 5,632 | 4 | 8.1 | 42.8 | 74.1 | 107.8 | 74 |
| 5,632 | 8 | 7.5 | 67.8 | 111.4 | 143.1 | 74 |

With one warp a center, the scan was about 9.5 of the 19.7 µs for 88 atoms:
the warp walks the 27 cells of the stencil, 9 columns of 3, one after the
other, and a center waits for every step of it. With four warps the
columns go to the warps in turns (3, 2, 2 and 2 of them) and the scan takes
about 3.3 µs. For 5,632 atoms the blocks already fill the device, and the
double precision pass and the ranking take most of the kernel. The kernel
in four warps: every warp appends its batches of 32 atoms to the one buffer
of candidates of the block through a count in shared memory (an atomic add
a batch), the double precision pass and the ranking take the candidates 128
at a time, and the nearest distance and the status bits of a center come
from the warps through shared memory and two barriers that also reduce. A
slot follows from the ranks alone, so the order in which the warps fill the
buffer does not matter: in this probe 2, 3, 4, 8, 9 and 16 warps gave the
list of one warp, bit for bit, on all five frames.

**What a skin would save.** On the same probe, the candidates stored by a
search with a scan radius of rcut + s are what the refresh of a list with a
skin of s reads back; the refresh moves each image by the shifts of its two
atoms, measures the candidates in double precision and ranks them as a
search does. Four warps a center (µs a launch, and the host's packing of
the input of a search, binning included):

| atoms | fresh search to rcut | read back: rcut | rcut + 1 Å | rcut + 2 Å | candidates: rcut | + 1 Å | + 2 Å | host packing |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 88 | 12.8 | 8.3 | 11.9 | 13.4 | 74 | 119 | 177 | 1.7 |
| 169 | 12.3 | 9.4 | 13.1 | 14.9 | 71 | 112 | 163 | 3.5 |
| 224 | 13.2 | 10.4 | 14.3 | 15.3 | 78 | 124 | 187 | 5.9 |
| 704 | 19.4 | 15.7 | 18.6 | 28.6 | 74 | 119 | 177 | 16.2 |
| 5,632 | 107.8 | 82.5 | 100.4 | 157.9 | 74 | 119 | 177 | 153.9 |

A skin of zero is the bound: about 3 to 4.5 µs a step for the frames of the
runs (one pass of the probe, whose stops scatter by a microsecond or two). A
skin that lasts (1 Å: 21 to 125 steps) reads back 112 to 124 candidates
where a search keeps 71 to 78, and that takes 11.9 to 14.3 µs against 12.3
to 13.2 for a fresh search, within the scatter of the probe. A skin would
also spare most of the host's packing, 1.7 to 5.9 µs at these sizes (a
refresh still needs the positions on the device). Thousands of atoms are
another case: for 5,632 atoms a 1 Å skin reads back in 100.4 µs against
107.8 for a fresh search, and the packing is 154 µs here (125 to 167 over
the runs of the probe), more than the kernel. There a skin, or a faster
packing, could save about 5% of an evaluation of 2,770 µs: the host part of
the search is what to look at for large systems.

**Warps per center.** A probe evaluator whose search takes its warps per
center from the environment, and the ranking loop unrolled by 4 or 8
(`WARPS:UNROLL`). The HAP frames with the trained se_e2_a model in float32
with tables, the 88-atom frame replicated, and the water frames with the
water model in float64; the least of five rounds, the order of the
configurations reversed every other round (µs an evaluation;
`hap-search-warps-rtx4090-2026-11-20.json`, with the medians):

| case | 1 | 2 | 4 | 8 | 4:4 | 2:4 | 4:8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HAP 88 | 156.0 | 149.7 | 149.3 | 147.4 | 150.9 | 150.3 | 150.3 |
| HAP 169 | 193.2 | 187.0 | 185.1 | 185.3 | 184.6 | 186.7 | 185.9 |
| HAP 224 | 215.7 | 210.3 | 209.3 | 208.5 | 207.9 | 211.8 | 209.1 |
| HAP 704 | 527.1 | 522.2 | 518.0 | 523.8 | 516.3 | 517.3 | 516.8 |
| HAP 5632 | 2776.2 | 2763.0 | 2768.4 | 2778.7 | 2778.8 | 2762.0 | 2762.8 |
| water 192 f64 | 1255.9 | 1239.5 | 1232.6 | 1236.9 | 1238.7 | 1242.7 | 1238.5 |
| water 1536 f64 | 8435.3 | 8462.7 | 8440.8 | 8445.7 | 8431.9 | 8449.4 | 8440.4 |

Four warps were the fastest or within the scatter of the fastest
throughout; eight were within the scatter of four up to 224 atoms and for
the water frames, slower from 704 HAP atoms on, and cost the kernel a third
more for 5,632 atoms. Unrolling the ranking loop changed nothing beyond the
scatter, and is not done.

**Before and after.** The evaluator of 475cafb and of this change, the one
before twice a round (`before again`: what the machine adds), five rounds,
the least of each (µs an evaluation; `hap-eval-search-warps-rtx4090-2026-11-20.json`,
with the medians):

| case | before | after | before again | after − before |
|---|---:|---:|---:|---:|
| HAP 88, float32 with tables | 155.0 | 149.8 | 157.4 | -5.3 |
| HAP 169, float32 with tables | 192.6 | 188.2 | 193.3 | -4.5 |
| HAP 224, float32 with tables | 217.4 | 210.2 | 216.1 | -7.3 |
| HAP 88, float64 with tables | 411.8 | 407.4 | 413.3 | -4.4 |
| HAP 169, float64 with tables | 720.3 | 715.5 | 720.1 | -4.7 |
| HAP 224, float64 with tables | 748.6 | 749.1 | 752.4 | +0.5 |
| HAP 704, float32 with tables | 524.8 | 521.4 | 525.9 | -3.4 |
| HAP 5,632, float32 with tables | 2,797.8 | 2,810.9 | 2,784.2 | +13.1 |
| water 192, float64 | 1,259.9 | 1,243.1 | 1,258.7 | -16.8 |
| water 1,536, float64 | 8,482.0 | 8,494.6 | 8,490.0 | +12.6 |
| water 12,288, float64 | 66,071.3 | 66,165.2 | 66,112.5 | +93.9 |

From 1,536 atoms on the differences are a fraction of a percent either way.
For 5,632 HAP atoms and 1,536 waters they are about what `before again`
differs from `before` (+13.1 against 13.6 µs, +12.6 against 8.0; the
medians: 2,836 → 2,819 µs and 8,500 → 8,499). For 12,288 waters the build
after was 93.9 µs (0.14%) slower, more than the 41.2 by which the build
before differed from itself, and 27.9 slower by the medians (66,175 →
66,202); that was not measured further. The 224-atom frame in float64 reads
757.1 → 750.6 µs by the medians. The same two builds again later in the
session, nine rounds on the HAP cases only, gave less for 88 atoms (µs;
`second_run` in the record):

| case | before | after | before again | after − before |
|---|---:|---:|---:|---:|
| HAP 88, float32 with tables | 152.5 | 150.3 | 152.5 | -2.3 |
| HAP 169, float32 with tables | 189.7 | 186.1 | 190.8 | -3.7 |
| HAP 224, float32 with tables | 213.8 | 208.9 | 214.2 | -4.9 |
| HAP 88, float64 with tables | 403.9 | 400.1 | 407.9 | -3.8 |
| HAP 704, float32 with tables | 522.6 | 517.2 | 522.8 | -5.4 |
| HAP 5,632, float32 with tables | 2,769.3 | 2,766.6 | 2,770.9 | -2.7 |

So an evaluation of 88 to 704 atoms is 2 to 7 µs shorter, less than the 7
µs the kernel lost for 88 atoms; why the evaluation keeps less of it was
not measured.

The engine: NVT molecular dynamics (CSVR, 300 K, τ 50 fs) of the three
frames, 20,000 steps of 0.5 fs, the same model in float32 with tables, the
engines of 475cafb and of this change alternated in five rounds, the least
of each, beside evaluations of the same frames by the two evaluators (µs;
`hap-md-step-search-warps-rtx4090-2026-11-20.json`):

| frame | per | before | after | before again |
|---|---|---:|---:|---:|
| 88 atoms | step | 170.5 | 165.8 | 170.4 |
| 88 atoms | evaluation | 155.6 | 154.7 | 154.6 |
| 169 atoms | step | 241.7 | 232.2 | 222.9 |
| 169 atoms | evaluation | 193.2 | 188.7 | 193.2 |
| 224 atoms | step | 239.4 | 234.3 | 238.3 |
| 224 atoms | evaluation | 214.4 | 209.4 | 217.4 |
| 169 atoms, again in seven rounds | step | 243.8 | 221.7 | 239.8 |
| 169 atoms, again in seven rounds | evaluation | 191.5 | 187.0 | 192.7 |
| 88 atoms, again in seven rounds | step | 162.2 | 159.8 | 163.4 |
| 88 atoms, again in seven rounds | evaluation | 152.2 | 150.2 | 152.7 |

A step of 88 atoms is 2.4 to 4.7 µs shorter and one of 224 atoms 5.1. The
step of the 169-atom frame sits near 222 or near 240 µs from run to run with
either build (the build before: 241.7 and 222.9 in the first run), so its
steps are not compared here; why it has two levels was not looked into. The
evaluations of 88 atoms beside the steps of the first run differed by
nothing, those of the rerun by 2 µs.

**Tests and mutants.** `every_count_of_warps_gives_the_list_of_one`
(`gpu/tests/device.rs`) searches seven frames (cubic, skewed, narrower than
the cutoff, 392 cells, truncated, open, and open with a buffer of seven so
that some centers take the serial path) with 1, 2, 3, 4, 5, 8 and 16 warps,
and asks for the list of one warp bit for bit; then systems searched
together with a tight selection, coincident pairs among centers whose
candidates take more than one warp, and the status of every center; then
forty atom types, more than a warp has lanes, of which only the last eight,
counted by other warps, are truncated. A unit test bounds the shared memory
of a block at its largest buffer for 1 to 16 warps and 1 to 40 types. Of 14
mutants of the kernel and its launch (the columns dealt by four whatever the
warps, the coincident bit or the truncation bit or the nearest distance from
warp 0 only, no broadcast of a batch's place, no barrier before the count of
candidates, the double precision pass a warp's own, counts not atomic, a
block for several centers, warps clamped to 32 or from 0, shared memory
without the warps' minima, the capacity bound for one warp), all 14 fail a
test; the truncation bit from warp 0 only failed none until the case of forty
types was added.

# The sums of the force kernels, and what the fitting nets wait for, 2026-11-20

The section below left the device's time over an evaluation of the 88-atom
frame by stage: the fitting nets 62 of 158 µs, the search 28, the forces 25.
This one takes the fitting nets and the force kernel apart, kernel by kernel
and step by step. The fitting nets gave nothing worth changing, and what was
tried there is here with its numbers. The force kernels did work in double
precision for one lane of a warp in two places, which is gone, and the sums
of their blocks had a fault that no model in use can meet, which is mended.
The engine's step was timed beside an evaluation of the same frame, and the
trainer, which shares the force kernels, beside the build before.

**A kernel taken apart in place.** A probe build gives a kernel one more
argument, the number of its steps after which it returns, and launches it
four more times with that argument before its own launch. The results of the
pass stay what they are, the kernels after it read what they read otherwise,
and the time added is four times the kernel's up to that step, with the
device behind the host as in any pass. (A stop fixed at compile time would
not do: the compiler drops the loops whose results nothing reads.) The least
of five or seven rounds; a step is the difference of two stops, and one
under a microsecond is within what the rounds scatter.

**The fitting nets.** 1,600 inputs, three hidden layers of 240 and an
output, in float32: three kernels a pass, the first hidden layer forward in
tiles, the other layers forward and backward with two rows a block, the
first layer backward in tiles. By the rows of a pass (the HAP frames of 54,
88, 169 and 224 atoms; the nets alone, 400 passes one after the other, µs):

| step | 54 rows | 88 rows | 169 rows | 224 rows |
|---|---:|---:|---:|---:|
| first layer forward, in tiles: launched and returning at once | 2.2 | 2.5 | 2.4 | 2.0 |
| its weights and inputs to shared memory | 2.0 | 2.7 | 4.0 | 4.2 |
| its sums | 3.7 | 6.3 | 10.1 | 13.3 |
| the other layers, two rows a block: launched and returning at once | 2.4 | 2.4 | 1.9 | 1.9 |
| the first layer from its partial sums | 4.4 | 4.5 | 4.7 | 4.8 |
| second hidden layer forward | 5.8 | 5.8 | 6.1 | 7.2 |
| third hidden layer forward | 5.5 | 5.5 | 5.9 | 6.7 |
| output layer and seeds | 0.6 | 0.7 | 0.7 | 0.8 |
| third hidden layer backward | 5.1 | 5.0 | 5.4 | 6.3 |
| second hidden layer backward, and the end of the kernel | 5.2 | 5.4 | 5.2 | 5.8 |
| first layer backward, in tiles: launched and returning at once | 2.4 | 2.6 | 2.8 | 2.5 |
| its weights and adjoints to shared memory | 3.5 | 5.8 | 8.6 | 10.0 |
| its sums | 8.2 | 10.5 | 19.4 | 25.1 |
| the three kernels | 51.0 | 59.7 | 77.3 | 90.7 |
| a pass of the nets | 49.4 | 57.8 | 77.9 | 90.5 |

- The kernel of the other layers takes 29 to 34 µs whatever the rows. Its
  block reads the weights of the second and third hidden layers once going
  forward and once going backward, 230 KB each time, at 5 to 7 µs a time.
- The two kernels of the first layer grow with the rows: 22 µs for 54 rows
  and 57 for 224, 45% of the pass and 63%.
- Launching the three kernels is 6 to 8 µs of a pass: 2 to 3 µs for each, as
  long as a kernel that returns at once takes.

Three things were tried there on probe builds, and none was taken, each
being worth less than 3 µs of an evaluation of 150:

- The sums of the middle kernel split among two, three or four sets of
  threads in a block (512 to 1,024 threads), each set a share of the inputs
  of every neuron, the shares added up through shared memory. With two sets
  a pass of 54 rows took 47.8 µs where it takes 49.9, one of 88 rows 56.3
  where it takes 58.3, and one of 224 rows 91.8 where it takes 91.0; three
  and four sets were no better than two, and a hidden layer still took 5 µs
  with a quarter of the reads to a thread. (The same code with one set took
  1.2 to 1.9 µs longer than the kernel as it is.)
- Four rows a warp in the forward tile kernel instead of two, so that a
  weight read from shared memory serves four products: the kernel went from
  11.6 to 9.8 µs at 88 rows, from 16.5 to 15.3 at 169 and from 19.7 to 17.2
  at 224, and stayed at 8.1 to 8.2 at 54; a pass of 88 rows from 58.3 to
  56.7 µs and one of 224 from 90.8 to 88.7. With chunks of 64 rows instead
  of 32 the kernel took 15.9 µs at 224 rows but 12.9 at 88.
- Loads of 16 bytes a lane, for a quarter of the load instructions over the
  same weights: a warp alone took 35 ns for such a load where one of 4 bytes
  takes 11.5, so 8.8 ns a weight instead of 11.5 (a probe of `load_costs` in
  `gpu/tests/costs.rs`; its modes were not kept). The first layer copies its
  weights in 2 to 10 µs a kernel, so less than a quarter of that was to be
  had.

**The force kernel of se_e2_a.** In place in an evaluation in float32,
before → after the two changes below (probe builds, µs):

| step | 88 atoms | 169 atoms | 224 atoms |
|---|---:|---:|---:|
| launched and returning at once | 2.0 → 2.1 | 2.2 → 2.5 | 2.3 → 2.4 |
| B of the atom to shared memory, and its adjoint | 2.0 → 3.1 | 2.4 → 2.1 | 2.8 → 2.0 |
| the sums of the slots, a warp a slot | 13.4 → 9.6 | 16.5 → 14.0 | 16.5 → 13.4 |
| the Jacobians and the forces on the neighbors, a thread a slot | 1.4 → 2.5 | 3.4 → 2.9 | 2.9 → 4.8 |
| the sums of the block | 4.3 → 1.5 | 7.9 → 2.6 | 8.8 → 2.3 |
| the kernel | 23.2 → 18.7 | 32.4 → 24.1 | 33.3 → 24.9 |

- *The sums of the slots.* A warp sums dE/d(env row) of one neighbor slot
  over the outputs of the embedding in the precision of the networks. Its
  first lane then turned the five sums into double and scaled them: eleven
  instructions in double precision a slot for one lane's numbers, and an
  instruction in double precision takes the multiprocessor as long with one
  lane at work as with thirty-two. The five sums now go to shared memory as
  they are, and the thread that has the slot in the next step, where every
  lane has one, turns them into double with the same operations; that step
  took up to 2 µs more for it. The force kernel of se_e2_r did the same with
  one sum a slot.
- *The sums of a block.* Every thread holds what its slots gave to the force
  on the block's atom and to the nine entries of its virial, in double.
  Every warp reduced its twelve over its lanes, five additions each with
  half the lanes at work, then a quarter, and so on: sixty additions in
  double precision a warp, 480 for a block of eight warps. The threads now
  put down what they hold in shared memory (twelve doubles a thread), and a
  warp for each sum adds up the warps lane by lane and reduces its lanes
  once: 156 additions for a block of eight warps. This is in the force
  kernels of all three descriptors, for the passes named below.

What is left of the slots' step, 10 to 14 µs, was not taken apart further.

**A fault in the sums of a block, mended.** With these changes the trainer,
which still has every warp reduce its own sums, no longer agreed with the
potential on one of its tests: a frame whose first atom has 260 neighbors,
under se_e3, whose force kernel has blocks of 256 threads and so gives four
of them two neighbors each. Against the CPU reference the sums across the
warps were right and those of every warp alone were not: the force on that
atom was 4e-9 to 1.5e-8 eV/Å off for 257, 260 and 300 neighbors, the virial
about 1e-8 eV, in every run alike, and the forces of the frame added up to
that much instead of to nothing, so what the block gave its neighbors was
right and its own sums were not. A barrier of the block before the warps
reduce their sums sets it right; the sums across the warps have one where
the threads have put theirs down. The cause was not found: a barrier of the
warp in that place set it right for one model and not for another, and a
barrier of the block between the reductions and the atomics that add them up
did not. The force kernels of se_e2_a and se_e2_r summed rightly without a
barrier in the three cases tried (shells of 130, 137 and 200 neighbors,
blocks of 128 and 256 threads), and the float64 forces of the HAP models,
whose 212 slots are more than a block of 128 has threads, are within 7.8e-11
eV/Å of DeePMD-kit's on the 83 frames of the sealed test (with tables, the
section of 2026-11-18). No model on disk has an se_e3 descriptor with more
than 138 slots, so none could give a thread of that kernel two neighbors.
Every force kernel now has the barrier, the trainer's too.

**Which passes take the sums across the warps.** The room for them is twelve
doubles a thread, 12 KB for a block of 128 threads and 24 KB for one of 256,
beside what a block had. One build with every warp reducing its own in every
pass against the sums across the warps in every pass that has room for them
(`--block-sums 0` and `--block-sums 1000000`), the widths of the blocks as
they are below, alternated in five rounds, the least of each: what the sums
across the warps add to a whole evaluation or take off it, by the centers of
the pass (its atoms times its models):

| centers | the pass | se_e2_a, float32 | se_e2_r, float32 | se_e3, float32 | se_e2_a, float64 | se_e2_r, float64 | se_e3, float64 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 54 | a frame of 54 atoms | −1.9% | −2.0% |  | −1.0% | −1.2% |  |
| 88 | a frame of 88 atoms | −1.8% | −1.8% | −2.7% | −0.9% | −1.7% | +3.1% |
| 108 | two frames of 54 | −1.8% | −1.4% |  | −0.6% | −1.7% |  |
| 162 | three frames of 54 | −3.6% | −3.1% |  | −0.9% | −1.7% |  |
| 169 | a frame of 169 atoms | −3.7% | −4.1% | −2.1% | −0.8% | −3.1% | −1.3% |
| 176 | two frames of 88 | −3.2% | −3.3% | −2.2% | −1.1% | −1.9% | −1.1% |
| 192 | water, a frame of 192 atoms | −2.6% | −4.0% | −1.0% | −1.4% | −1.8% | −1.5% |
| 216 | four frames of 54 | −2.9% | −3.3% |  | −0.6% | −2.1% |  |
| 224 | a frame of 224 atoms | −2.5% | −3.4% | −2.7% | −0.8% | −1.6% | −1.4% |
| 264 | three frames of 88 | −3.7% | −4.0% | −3.2% | −0.3% | −0.7% | −1.3% |
| 270 | five frames of 54 | −2.7% | −3.6% |  | −0.2% | −1.0% |  |
| 352 | four frames of 88 | −3.1% | −4.7% | −2.5% | −0.6% | −0.6% | −1.7% |
| 352 | four models, a frame of 88 atoms | −2.2% |  |  | −0.6% |  |  |
| 528 | six frames of 88 | +3.7% | −2.4% | −3.9% | +2.8% | −0.8% | −1.2% |
| 896 | four models, a frame of 224 atoms | +2.2% |  |  | 0.0% |  |  |
| 1,536 | water, a frame of 1,536 atoms | +0.5% | −3.2% | −2.8% | −1.0% | −0.1% | −0.7% |
| 12,288 | water, a frame of 12,288 atoms | +0.1% | 0.0% |  | 0.0% | −0.2% |  |

A setting against itself is 0.3% apart in the middle and 1.3% at most (5 of
94 cases over 1%). With an se_e3 descriptor beside the se_e2_a one, at 88,
169 and 224 atoms: −1.7%, −3.8%, −3.6% in float32 and −0.5%, −2.3%, −0.9% in
float64.

In float32 the sums across the warps take 1.0% to 4.7% off every evaluation
of up to 352 centers, whatever the descriptor. Past that the descriptors
part: with se_e2_a an evaluation of 528 centers takes 3.7% longer and a
committee of 896 2.2% (at 1,536 and 12,288 within 0.5%), with se_e2_r or
se_e3 one of 528 or 1,536 centers 2.4% to 3.9% shorter. In float64 the gain
is smaller: 0.2% to 1.4% with se_e2_a up to 352 centers and 2.8% more at
528, 0.6% to 3.1% with se_e2_r up to 528, and 0.7% to 1.7% with se_e3 from
169 centers on. At 88 centers an evaluation with se_e3 in float64 took 3.1%
longer with them, and 1.4% as a batch of one frame. So a pass with an
se_e2_a descriptor forms the sums across the warps up to 352 centers
(`native::gpu_model::BLOCK_SUMS_CENTERS`), passes with se_e2_r and se_e3 at
any size, and every pass only where its block has room within 48 KB
(`set_block_sums_centers` and the evaluator's `--block-sums` move the limit
for every descriptor; `block_sums_shared` says which way a pass took).
Nothing was measured between 352 and 528 centers, so the limit lies
somewhere in between.

**The widths of the blocks again.** The limits of the wider blocks of the
two-body force kernels (the section below) were found while every warp
reduced its own sums, which a block of eight warps did twice as much of as
one of four. With the sums as they are now, blocks of 128 threads for every
pass against blocks of 256 for every pass, in five rounds:

| centers | the pass | se_e2_a, float32 | se_e2_r, float32 | se_e2_a, float64 | se_e2_r, float64 |
|---:|---|---:|---:|---:|---:|
| 54 | a frame of 54 atoms | −4.9% | −5.6% | −0.9% | −4.2% |
| 88 | a frame of 88 atoms | −5.0% | −5.0% | −0.7% | −4.4% |
| 108 | two frames of 54 | −4.3% | −4.3% | −0.7% | −3.5% |
| 162 | three frames of 54 | −1.7% | −4.9% | −0.3% | −2.5% |
| 169 | a frame of 169 atoms | −3.2% | −4.5% | −0.3% | −2.7% |
| 176 | two frames of 88 | −2.6% | −4.4% | −1.1% | −1.6% |
| 192 | water, a frame of 192 atoms | −2.7% | −4.2% | −0.1% | −2.1% |
| 216 | four frames of 54 | −1.2% | −3.6% | +0.5% | −2.5% |
| 224 | a frame of 224 atoms | −3.1% | −4.2% | −0.2% | −1.8% |
| 264 | three frames of 88 | −1.3% | −3.9% | −1.0% | −0.5% |
| 270 | five frames of 54 | −2.8% | −3.9% | +0.2% | +0.2% |
| 352 | four frames of 88 | −2.0% | −2.2% | −0.3% | −0.8% |
| 352 | four models, a frame of 88 atoms | −2.1% |  | +4.0% |  |
| 528 | six frames of 88 | +2.9% | +1.6% | −0.7% | +1.7% |
| 896 | four models, a frame of 224 atoms | +2.9% |  | −1.5% |  |
| 1,536 | water, a frame of 1,536 atoms | −0.4% | +1.2% | +0.5% | +0.6% |
| 12,288 | water, a frame of 12,288 atoms | +2.8% | +3.2% | +0.6% | +0.1% |

A setting against itself is 0.3% apart in the middle and 2.3% at most (9 of
74 cases over 1%). With an se_e3 descriptor beside the se_e2_a one, at 88,
169 and 224 atoms: −4.0%, −1.8%, −2.4% in float32 and −0.7%, +0.2%, +0.9% in
float64.

In float32 the wider blocks take 4% to 6% off an evaluation at 54 to 108
centers and 1.2% to 5% from 162 to 352, where the gain had ended at 224; at
528, 896 and 12,288 centers they add 1.6% to 3.2%, at 1,536 they are within
1.2%. In float64 they take 1.6% to 4.4% off an evaluation with se_e2_r up to
224 centers (up to 108 before), less than 1% at 264 to 352, and add 1.7% at
528; one with se_e2_a stays within 1.7% at every size but one, a committee
of 352 centers, 4.0% longer. The wider blocks therefore go to float32 passes
of up to 352 centers and float64 passes of up to 224
(`WIDE_FORCE_CENTERS_F32` and `WIDE_FORCE_CENTERS_F64`, 224 and 108 before).
Both limits of 352 come from the same grid of passes, with nothing between
352 and 528. This comparison was run before the barrier of the section
above: in its se_e2_a passes of more than 352 centers every warp reduced its
own sums without it, in both widths.

**Measured.** The build before (a798b12) and this one with `--compress 0.5`,
alternated in seven rounds in one session, the least of each kept, before →
after. Trained HAP models on frames of the sealed test, microseconds for
energy, forces and virial (`--bench 3000`):

| frame | precision | se_e2_a to 6 Å | se_e2_r to 6 Å | se_e3 to 4 Å | se_e2_a to 6 Å with se_e3 to 4 Å |
|---|---|---:|---:|---:|---:|
| 88 atoms | float32 | 150 → 145 | 113 → 111 | 128 → 127 | 212 → 205 |
| 88 atoms | float64 | 382 → 378 | 181 → 179 | 207 → 208 | 498 → 494 |
| 169 atoms | float32 | 185 → 177 | 142 → 136 | 163 → 158 | 270 → 253 |
| 169 atoms | float64 | 678 → 669 | 281 → 271 | 320 → 315 | 835 → 824 |
| 224 atoms | float32 | 208 → 200 | 165 → 158 | 188 → 183 | 290 → 276 |
| 224 atoms | float64 | 709 → 701 | 309 → 301 | 368 → 365 | 893 → 883 |

Other passes:

| pass | centers | float32 | float64 |
|---|---:|---:|---:|
| four se_e2_a models, a frame of 88 atoms | 352 | 346 → 334 | 1,042 → 1,028 |
| four se_e2_a models, a frame of 224 atoms | 896 | 612 → 611 | 2,602 → 2,578 |
| se_e2_a, one frame of 88 atoms | 88 | 151 → 146 | 383 → 379 |
| se_e2_a, two frames of 88 atoms | 176 | 187 → 178 | 674 → 664 |
| se_e2_a, three frames of 88 atoms | 264 | 240 → 228 | 986 → 975 |
| se_e2_a, four frames of 88 atoms | 352 | 277 → 266 | 1,043 → 1,026 |
| se_e2_a, six frames of 88 atoms | 528 | 398 → 394 | 1,623 → 1,608 |
| se_e3, one frame of 88 atoms | 88 | 127 → 128 | 206 → 207 |
| se_e3, two frames of 88 atoms | 176 | 166 → 161 | 324 → 320 |
| se_e3, three frames of 88 atoms | 264 | 202 → 193 | 463 → 455 |
| se_e3, four frames of 88 atoms | 352 | 212 → 203 | 491 → 482 |
| se_e3, six frames of 88 atoms | 528 | 312 → 294 | 732 → 721 |
| se_e2_a, water, a frame of 192 atoms | 192 | 212 → 205 | 704 → 697 |
| se_e2_a, water, a frame of 1,536 atoms | 1,536 | 891 → 883 | 4,279 → 4,243 |
| se_e2_a, water, a frame of 12,288 atoms | 12,288 | 5,115 → 5,005 | 32,516 → 32,261 |
| se_e2_r, water, a frame of 192 atoms | 192 | 172 → 167 | 315 → 305 |
| se_e2_r, water, a frame of 1,536 atoms | 1,536 | 733 → 714 | 1,802 → 1,777 |
| se_e2_r, water, a frame of 12,288 atoms | 12,288 | 3,929 → 3,863 | 13,218 → 13,103 |
| se_e3, water, a frame of 192 atoms | 192 | 419 → 413 | 1,075 → 1,071 |
| se_e3, water, a frame of 1,536 atoms | 1,536 | 2,201 → 2,164 | 6,275 → 6,253 |

The build before against itself: 0.21% apart in the middle and 0.97% at
most, none of the 64 cases over 1%. In float32 the passes are 3.4% shorter
in the middle and up to 6.4%; the pipeline's path, se_e2_a with tables, went
from 150 to 145 µs at 88 atoms (3.3%), from 185 to 177 at 169 (4.7%) and
from 208 to 200 at 224 (3.9%). se_e3 on the 88-atom frame came out 1.2%
shorter alone and 0.6% longer as a batch of one, where the build before
differed from itself by 1.0% and 0.3% there: no change that this run can
tell. In float64 the passes are 1.1% shorter in the middle and up to 3.3%,
and se_e3 at 88 atoms came out 0.9% longer alone, where the build before
differed from itself by 0.6% (0.1% longer as a batch of one); the sums
across the warps cost it 1.4% to 3.1% there in the comparison of the two
ways (above). Past the limits a pass keeps the sums of the slots, the
barrier and, with se_e2_r or se_e3, the sums across the warps: a committee
of 896 centers in float32 is 0.1% shorter, six frames of 88 atoms with
se_e2_a 1.1%, the water frames of 1,536 and 12,288 atoms 0.9% to 2.5% in
float32 and 0.3% to 1.4% in float64. The build after with the limits of the
build before (`--block-sums 0`, `--wide-forces 224` in float32 and `108` in
float64; the third column of the record) has the sums of the slots in the
precision of the networks and the barrier as its only differences: it took
1.0% off a float32 evaluation in the middle (from 1.0% longer, with se_e2_r
at 88 atoms, to 2.0% shorter) and 0.3% off a float64 one (from 0.7% longer
to 1.4% shorter). Most of the gain is the sums across the warps and the
wider blocks up to the new limits. No frame left its tables.

**The engine's step.** `omnimd-mlmd run` with the trained se_e2_a model of
the evaluator's benchmarks in float32 with tables (`compress_min_distance`
0.5): NVT (CSVR, 300 K, τ 50 fs) of a frame of the sealed test, 20,000 steps
of 0.5 fs, a sample every 50 steps; the engine prints the time of its steps,
model loading not included. The engine of a61ca3e (the evaluation code of
a798b12) and this one, alternated in five rounds with the two evaluators
beside them on the same frames and model (`--bench 3000`), the least of
each, µs:

| frame | a step | an evaluation |
|---|---:|---:|
| 88 atoms | 161.0 → 157.4 | 152.5 → 146.6 |
| 169 atoms | 216.2 → 199.2 | 186.7 → 178.5 |
| 224 atoms | 222.6 → 217.4 | 210.6 → 200.9 |

The engine before against itself came out 0.4% to 1.1% apart, the evaluator
0.2% to 0.6%. At 88 and 224 atoms a step took 8 to 17 µs more than an
evaluation of its frame (the integrator, the thermostat, the samples), and
the steps gained 3.6 and 5.2 µs where the evaluations gained 5.9 and 9.7. At
169 atoms a step took 30 µs more than an evaluation before and 21 after, so
the step went 17 µs shorter where the evaluation went 8 µs shorter; that was
not looked into.

**The trainer.** omnimd-native-gpu-train launches the same force kernels: it
has the sums of the slots in the precision of the networks and the barrier,
and every warp still reduces its own sums. 10,000 steps of the first-pass
dataset of HAP iteration 0 (171 frames of 54 to 224 atoms, a frame a step)
in float32, the two builds alternated in three rounds, the best rate of
each: with the se_e2_a pipeline model 588.1 → 594.4 steps a second (1.1%
more, where the build before differed from itself by 0.3%), with an se_e3
model to 4 Å 1,405.5 → 1,406.0 (the same; 0.1%). The losses at the end of a
run differ from run to run within either build (the forces are added up by
atomics in no fixed order), so they say nothing of the change.

**Host memory that kernels read and write.** An evaluation uploads a few
kilobytes before its first kernel and copies a few back after its last. A
probe (not kept) had a kernel read its input from host memory that is pinned
and mapped to the device, and write its output there, with about 30 µs of
device work between as in an evaluation: a cycle of 8 KB each way took 51.7
µs with the two copies, 51.7 with the input read from host memory, 49.2 with
the output written to it and a wait for the stream, 46.4 with both; with 32
KB each way 55.9, 58.4, 50.8 and 52.4. (With no work between, 22.3 against
11.1 µs.) The force kernels add to their results with atomics, which such
memory is not for, so the results would need a kernel of their own to carry
them over: not done.

**What this costs.**

- A float64 se_e2_a model needs 40 bytes of shared memory a neighbor slot in
  the force kernel where it needed 32: at an embedding width of 100 the
  check at load refuses one from 1,067 slots, where it did from 1,334. No
  model on disk is refused for it (all 150 files checked in float64 with
  both builds; three of them are empty and refused by both). A float32 model
  needs 20 bytes a slot where it needed 32.
- The sums across the warps take 12 or 24 KB of shared memory a block more,
  which costs a pass with many blocks: with se_e2_a they add 2.8% to 3.7% at
  528 centers, and so stop at 352, a limit measured on one card with nothing
  measured between 352 and 528.
- se_e3 in float64 at 88 centers: 1.4% to 3.1% longer with the sums across
  the warps, which it takes at every size.
- The barrier is in every force kernel, also where every warp reduces its
  own sums: se_e2_a passes past 352 centers, blocks without room, the
  trainer. The third column of the record (`after_by_warp`) and the
  trainer's runs above are what it costs.
- The forces and the virial of a pass formed either way differ by the
  rounding of a block's sums, which are added in another order.

**Not done.** The levers of the fitting nets above, each under 3 µs; the 10
to 14 µs of the slots' step; the cause of the fault; a lower limit for se_e3
in float64; host memory mapped to the device; a check in every kernel that
its shared memory fits the size it is launched with, which would find what
memcheck does not (below).

**Tests** (on this machine; the GPU suites do not run in CI).
`the_sums_of_a_force_block_are_formed_either_way`
(`native/tests/gpu_parity.rs`) has every descriptor, a hybrid of three and
one that ends with se_e2_a (whose force kernel then adds to the virial of
the first), in both precisions, by both ways and with both widths, at the
limit of centers and one past it, as a potential starts and with one number
for all, for batches (also in chunks of a frame's atoms, which count a
chunk's centers) and a committee, a pass of one center with a limit of none
and of one, and a model whose 600 slots leave room for the sums of a block
of 128 threads and not for those of one of 256; each against the CPU
reference or with the same energies and forces within 1e-11.
`an_atom_with_more_neighbors_than_a_force_block_has_threads` has frames like
the one that showed the fault, for all three descriptors, by both ways and
both widths, against the CPU reference.
`few_centers_take_wider_blocks_in_the_two_body_force_kernels` has the new
limits. The trainer's test of an se_e3 atom with more neighbors than a
kernel has threads passes again.
`the_stages_of_an_evaluation_are_timed_by_the_device` bounds the device's
time of an evaluation by the time of the call, which waits for the last
mark, where it bounded it by the host's issuing: the last mark can reach the
device a while after the host issued it, and that bound failed now and then
in the mutation rounds under mutants that do not reach the test (once 0.88
ms on the device against 0.37 of issuing).

**Mutation rounds.** 63 mutants of this change (the kernels, the launches
and the trainer), each against the GPU parity tests or the trainer's: 56
were caught at once. Three more were caught once the test above had a pass
in chunks, a pass of one center and a hybrid ending with se_e2_a: a pass in
chunks counted by all its atoms, a limit of none read as one, and an se_e2_a
force kernel that forms its sums across the warps whenever it adds to the
virial, room or not (it wrote up to 24 KB past its block's shared memory,
and the device stopped on an illegal address). One gives the same forces and
no test can tell it: a launch that makes room for the sums and leaves the
kernel reducing warp by warp. Three were not caught by the tests: the
se_e2_a force kernel given 96 bytes of shared memory too few, and the
se_e2_r one given too little by its dE/dD, in the potential and in the
trainer. compute-sanitizer's memcheck over the force kernels' tests finds
the second, where the shortfall is 1,088 bytes (an embedding of 136 in
float64; 306 errors), and neither the third (at most 320 bytes in the
trainer's tests) nor the 96 bytes of se_e2_a: on this card it reported
accesses from 1,024 bytes past the size a block is launched with on, and
none before. Over the same tests of this change it finds no error. The first
round had counted three of these as caught by the old bound of the stages
test, which failed now and then under mutants that cannot reach it; run
twice again with the new bound, they came out as told here. The 50 mutants
of the earlier rounds on the wider blocks and the clock of the stages that
touch this code were run again, and all were caught. The 34 of the clock,
which only the stages test catches, were run once more with the new bound
and caught again: 33 by checks that the code as it is meets exactly (a stage
with no time, host times that do not add up to the call, means that are not
the means, times that are missing) and one by a device time of less than
half the issuing (in 20 runs of 20, at 0.30 to 0.41 of it); none by the
bound by the call.

Locally: the workspace 234 tests, the engine with `native-gpu` 93, the GPU
crates 192, each in release and debug; `cargo fmt --check` clean; the legacy
`cuda` features compile.

Files: `hap-eval-force-sums-rtx4090-2026-11-20.json` (before and after;
and `-r1`, a first run of five rounds an hour and a half earlier, every case
in it 8% to 14% longer for a reason not found, the cases bound by the device
too, and what the change takes off alike: 3.3% and 1.2% in the middle),
`hap-eval-block-sums-rtx4090-2026-11-19.json` (the two ways of the sums),
`hap-eval-force-blocks-rtx4090-2026-11-19-r2.json` (the two widths, with the
sums as they are now), `hap-md-step-rtx4090-2026-11-20.json` (the engine's
step) and `hap-train-force-sums-rtx4090-2026-11-20.json` (the trainer). The
probe builds and their outputs are not in the repository.

# Where an evaluation of a small frame goes, 2026-11-19

The pipeline's models evaluate frames of 54 to 224 atoms, and an evaluation
of one of them is a dozen device calls. The cost model of the device crate
said that twenty avoidable calls are a third of such a step, and the
sections below removed calls one after the other. This one set out to find
which calls of the production path (se_e2_a with tables, float32) are still
avoidable. It found none: the calls are not what such an evaluation waits
for. What it does wait for is named here stage by stage. Two things in it
are shorter now; what else was tried is listed with its numbers.

**Where the host is.** On a probe build of the code as it was (46a2f20) that
read the host's clock after the calls of every stage and waited for nothing,
an evaluation of the 88-atom frame took 166 µs: the host issued all its
calls in 40 µs (sorting the atoms 2, the neighbor search with its packing
and upload 13, the environment 6, the tables 3, the descriptor 3, the
fitting nets 9, the forces 3) and then waited 120 µs in the copy back. Two
rearrangements that have the host issue a kernel sooner left the evaluation
within 2 µs of where it was: the search issued before the results are
cleared instead of after, and the results copied back to pinned memory.

**Stage times by the device.** `--profile` waited for the device after every
stage, which times an evaluation that never runs otherwise: the device
stands still after every stage while the host issues the next. It now issues
a mark into the device's stream after the calls of each stage
(`omnimd_gpu::Device::mark`, a CUDA event) and reads the marks when the
results are back. `stages_ms.device` is the device's time between marks,
`stages_ms.issue` the host's time to issue the calls of each stage and its
mark, `wall` the whole. The 88-atom frame in float32 with tables, after the
two changes below (means of 3,000 evaluations, µs):

| | neighbors | environment | embedding | descriptor | fitting | forces | download | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `issue` | 21.7 | 10.2 | 7.0 | 13.6 | 25.2 | 8.0 | 87.0 | 172.8 |
| `device` | 28.3 | 7.0 | 13.0 | 13.0 | 66.6 | 25.5 | 13.2 | 166.6 |
| the device with one stage marked (a probe build) | 27.7 | 5.7 | 10.7 | 11.2 | 62.3 | 25.1 | 15.4 | 158.0 |

The marks are not free, and in a pass this short they show. One costs the
host 4.5 to 5 µs to issue on an idle device and the device 1.5
(`device_times_by_marks` in `gpu/tests/costs.rs`). With eight of them the
host issued for 86 µs before the copy back, where the probe build of the
paragraph above issued in 40, and the evaluation took 180 µs where it takes
151. The last row is the check: a probe build that issues two marks an
evaluation, at the two ends of one stage, run seven times with a stage each.
Those evaluations took 156 to 166 µs, and against them the stages of
`--profile` are 1.3, 2.2, 1.8 and 4.3 µs longer for the environment, the
embedding, the descriptor and the fitting nets, within a microsecond for the
search and the forces, and 2 µs shorter for the copy back. For the same
frame in float64 (388 µs) and the 224-atom frame in float32 (208 µs) the two
ways agree within a microsecond for every stage. So in this session
`--profile` was right to a microsecond a stage for those two passes of 208
and 388 µs, and put up to 5 µs too many on a stage of the shortest.

What the marks add goes by the pace of the host, and that is not the same
from one session to the next on this machine. The profile of an earlier
build of this change read 167 µs for the 88-atom frame on 2026-11-18; the
same binary read 178 and 188 today, beside 179 to 186 for the final one.
`device_times_by_marks` gave 4.9 µs to issue a mark on the 18th and 4.5,
4.4 and 5.2 in three runs today. The plain evaluation, the least of seven
rounds, was 150 µs in both runs of the record. A profile of a pass this
short is to be read with the plain time of the same session beside it.

Two of the device's numbers are not all its work. `device.neighbors` begins
at the first device call, and the device has nothing to do while the host
packs the input of the search: up to 22 µs of it, the host's
`issue.neighbors`, is that wait. And `device.download` ends at a mark the
host issues once it has the results, so it holds the host's return from its
wait. `StageTimes` says how the two clocks bound the device's waiting in the
other stages (here 3.7, 3.6, 4.3 and 16.5 µs at most for the environment,
the embedding, the descriptor and the fitting nets; the probe says less).
Stage times printed before today were taken with a wait after every stage
and are not comparable. All three frames in both precisions are in the
records.

**What the neighbor search waited for.** Stopped after each of its steps on
a probe build, by marks around the kernel on an idle device (which include
its launch call), the search of 88 atoms took 6.4 µs to launch and return at
once, 23.7 through the scan of the cells, 24.0 through the exact distances,
24.7 through the clearing of the slots and 29.8 in all: 17 of its 30 µs were
the scan. In the scan one warp walks the 27 cells of a center's stencil, and
it had the cell and the periodic image of each from a division for every
axis. A 32-bit division by a number the kernel is given takes a warp 27 ns,
as long as fourteen float multiply-adds one after the other
(`arithmetic_costs`), and a center waits for each. The scan now divides once
for an axis and steps from there, and it takes the cells along the last
axis, which lie one after the other in the grid with their atoms, as one run
of atoms: fewer bounds to read and fuller batches of 32 for the lanes. The
atoms come by in the order of the scan cell by cell, so the candidates, and
with them the list, are the same; the serial path for centers with more
neighbors than the buffer holds is as it was. By the same marks on probe
builds, the kernel for 5,632 atoms took 166 µs with the divisions, 142.5
with the steps and 116 with the runs too (to 4 Å: 87.7, 69.3 and 52.5), and
the one for 88 atoms 29.0 and then 22.7. A search with its packing and a
wait for the device, the kernel of 46a2f20 and this one alternated in five
rounds, the least of each (`neighbor_search_costs`): 37.1 → 29.6 µs for 88
atoms, 60.4 → 51.4 µs for 704 atoms, 313.0 → 279.9 µs for 5,632 atoms.

**Wider blocks for the forces of few centers.** The force kernel of se_e2_a
and se_e2_r is a block for a center, and one of its steps takes the neighbor
slots of the center a warp each, the warps of the block in turns: with 256
threads instead of 128, eight warps are through 212 slots in 27 turns
instead of 53. Whether a pass is sooner done with them goes by how many
blocks it has, by the precision and by the descriptor. One build with blocks
of 128 threads for every pass against blocks of 256 for every pass
(`--wide-forces 0` and `--wide-forces 1000000`), alternated in seven rounds,
the least of each: what the wider blocks add to a whole evaluation or take
off it. The centers are the atoms of the pass times its models, the blocks
of the kernel; the frames are HAP frames of the sealed test, the batches
copies of one frame:

| centers | the pass | se_e2_a, float32 | se_e2_r, float32 | se_e2_a, float64 | se_e2_r, float64 |
|---:|---|---:|---:|---:|---:|
| 54 | a frame of 54 atoms | −4.5% | −4.3% | −0.4% | −2.9% |
| 88 | a frame of 88 atoms | −4.8% | −3.9% | −0.3% | −3.0% |
| 108 | two frames of 54 | −3.9% | −3.2% | 0.0% | −3.4% |
| 162 | three frames of 54 | −2.1% | −1.0% | +0.6% | −0.7% |
| 169 | a frame of 169 atoms | −3.0% | −1.6% | +0.3% | −1.1% |
| 176 | two frames of 88 | −2.8% | −3.0% | +0.6% | −0.4% |
| 192 | water, a frame of 192 atoms | −1.2% |  | +0.1% |  |
| 216 | four frames of 54 | −1.6% | −0.6% | +0.6% | −0.5% |
| 224 | a frame of 224 atoms | −2.8% | −2.7% | +0.2% | −0.1% |
| 264 | three frames of 88 | +0.9% | −0.7% | +0.2% | +0.9% |
| 270 | five frames of 54 | +1.5% | 0.0% | +0.3% | +0.8% |
| 352 | four frames of 88 | +0.4% | −1.0% | +0.3% | +1.2% |
| 352 | four models, a frame of 88 atoms | −1.1% |  | +0.1% |  |
| 528 | six frames of 88 | +2.7% | +3.9% | +0.1% | +2.7% |
| 896 | four models, a frame of 224 atoms | +3.2% |  | −1.4% |  |
| 1,536 | water, a frame of 1,536 atoms | +0.9% |  | +0.4% |  |
| 12,288 | water, a frame of 12,288 atoms | +2.4% |  | +0.2% |  |

The narrow setting against itself is 0.3% apart in the middle and 1.7% at
most (3 of 68 cases over 1%). The frame of 88 atoms timed again through
`--bench-batches`: −4.9% and −4.8% in float32, −0.5% and −2.4% in float64.
With an se_e3 descriptor beside the se_e2_a one, at 88, 169 and 224 atoms:
−3.5%, −1.5% and −1.1% in float32 and −0.7%, +0.2% and +0.5% in float64.

In float32 the wider blocks take 3% to 5% off an evaluation up to 108
centers and 0.6% to 3% from 162 to 224. From 264 to 352 the two widths are
within 1.5% of each other, the sign going either way, which is what one
setting differs from itself by in the worst of these cases; from 528 on the
wider blocks add 1% to 4%. In float64 they take 2.4% to 3.4% off an
evaluation with se_e2_r up to 108 centers and less than 1.2% from 162 to
224, and add 1% to 3% from 264 on. An evaluation with se_e2_a in float64 is
within 0.6% either way at every size but one, the committee of 896 centers,
1.4% shorter. Why the gain ends where it does was not measured.

A float32 pass of at most 224 centers therefore takes the wider blocks, a
float64 pass of at most 108, and every other pass the blocks of 128
(`native::gpu_model::WIDE_FORCE_CENTERS_F32` and `WIDE_FORCE_CENTERS_F64`,
`set_wide_force_centers`, the evaluator's `--wide-forces`; `force_threads`
says which a pass took). The forces of the two widths differ by the rounding
of a center's sums: at most 8e-14 eV/Å on the HAP frames, where two runs of
one program differ by 3e-14, the forces being sums of atomic additions. The
energies are the same to the last bit. (Since the section of 2026-11-20
above the limits are 352 and 224, measured again with the sums of a block
formed across its warps.)

**Measured.** The build before (46a2f20) and this one with `--compress 0.5`,
alternated in seven rounds in one session, the least of each kept, before →
after. Trained HAP models on frames of the sealed test, microseconds for
energy, forces and virial (`--bench 3000`):

| frame | precision | se_e2_a to 6 Å | se_e2_r to 6 Å | se_e3 to 4 Å | se_e2_a to 6 Å with se_e3 to 4 Å |
|---|---|---:|---:|---:|---:|
| 88 atoms | float32 | 165 → 150 | 126 → 113 | 136 → 128 | 227 → 214 |
| 88 atoms | float64 | 396 → 385 | 197 → 183 | 215 → 206 | 502 → 494 |
| 169 atoms | float32 | 203 → 188 | 157 → 143 | 174 → 165 | 284 → 270 |
| 169 atoms | float64 | 698 → 689 | 293 → 285 | 334 → 323 | 854 → 844 |
| 224 atoms | float32 | 220 → 209 | 175 → 164 | 198 → 189 | 303 → 292 |
| 224 atoms | float64 | 728 → 719 | 319 → 312 | 379 → 368 | 906 → 899 |

The two changes apart, for the models with a two-body force kernel and the
passes that take the wider blocks: before → with the scan alone
(`--wide-forces 0`) → with the wider blocks too:

| frame | precision | se_e2_a to 6 Å | se_e2_r to 6 Å | se_e2_a to 6 Å with se_e3 to 4 Å |
|---|---|---:|---:|---:|
| 88 atoms | float32 | 165 → 157 → 150 | 126 → 118 → 113 | 227 → 221 → 214 |
| 88 atoms | float64 | 396 → 388 → 385 | 197 → 189 → 183 | 502 → 495 → 494 |
| 169 atoms | float32 | 203 → 192 → 188 | 157 → 147 → 143 | 284 → 275 → 270 |
| 224 atoms | float32 | 220 → 213 → 209 | 175 → 168 → 164 | 303 → 294 → 292 |

Other passes; where the blocks stay at 128 threads the scan is the one
difference:

| pass | centers | precision | threads of a force block | before → after |
|---|---:|---|---:|---:|
| four se_e2_a models, a frame of 88 atoms | 352 | float32 | 128 | 357 → 349 |
| four se_e2_a models, a frame of 88 atoms | 352 | float64 | 128 | 1,055 → 1,046 |
| four se_e2_a models, a frame of 224 atoms | 896 | float32 | 128 | 626 → 619 |
| four se_e2_a models, a frame of 224 atoms | 896 | float64 | 128 | 2,635 → 2,626 |
| se_e2_a, one frame of 88 atoms | 88 | float32 | 256 | 165 → 151 |
| se_e2_a, two frames of 88 atoms | 176 | float32 | 256 | 199 → 188 |
| se_e2_a, three frames of 88 atoms | 264 | float32 | 128 | 251 → 244 |
| se_e2_a, four frames of 88 atoms | 352 | float32 | 128 | 289 → 279 |
| se_e2_a, six frames of 88 atoms | 528 | float32 | 128 | 412 → 403 |
| se_e2_a, one frame of 88 atoms | 88 | float64 | 256 | 396 → 385 |
| se_e2_a, two frames of 88 atoms | 176 | float64 | 128 | 688 → 680 |
| se_e2_a, three frames of 88 atoms | 264 | float64 | 128 | 995 → 986 |
| se_e2_a, four frames of 88 atoms | 352 | float64 | 128 | 1,053 → 1,043 |
| se_e2_a, six frames of 88 atoms | 528 | float64 | 128 | 1,644 → 1,633 |
| se_e2_a, water, a frame of 192 atoms | 192 | float32 | 256 | 225 → 215 |
| se_e2_a, water, a frame of 192 atoms | 192 | float64 | 128 | 721 → 715 |
| se_e2_a, water, a frame of 1,536 atoms | 1,536 | float32 | 128 | 906 → 898 |
| se_e2_a, water, a frame of 1,536 atoms | 1,536 | float64 | 128 | 4,376 → 4,367 |
| se_e2_a, water, a frame of 12,288 atoms | 12,288 | float32 | 128 | 5,252 → 5,218 |
| se_e2_a, water, a frame of 12,288 atoms | 12,288 | float64 | 128 | 33,299 → 33,269 |

- The scan alone takes 6 to 11 µs off an evaluation of a HAP frame, for
  every model in both precisions (24 cases: 6.4 to 11.1 µs), se_e3 included,
  which has no two-body force kernel.
- The wider blocks take another 2 to 7 µs off a float32 evaluation of at
  most 224 centers (0.7% to 4.6%: most for the 88-atom frame, least beside
  an se_e3 descriptor at 224 atoms), and 5.9 µs (3.1%) off the float64
  evaluation of the 88-atom frame with se_e2_r. For se_e2_a in float64 at 88
  atoms the numbers say 3.4 µs (0.9%) here and 0.3% to 0.5% in the
  comparison of the widths above, which is what a setting differs from
  itself by.
- Together, for the pipeline's path (se_e2_a with tables in float32): 165 to
  150 µs at 88 atoms (9%), 203 to 188 at 169 (8%) and 220 to 209 at 224
  (5%). For se_e2_r: 126 to 113, 157 to 143 and 175 to 164 (10%, 9%, 7%).
- Batches, committees and float64 passes past the limits gain 6 to 12 µs,
  from the scan. The water frames of 1,536 and 12,288 atoms gain 8 to 9 and
  30 to 33 µs (in the first run 2 to 10 and 59 to 112): a fraction of a
  percent of 0.9 to 33 ms.
- The build before against itself is 0.19% apart in the middle and 0.82% at
  most; none of the 44 cases is over 1%. No frame left its tables.
- A first run of seven rounds (`-r1`) was made with the build before float64
  had a limit of its own. For the HAP frames in float32 its numbers are
  within 3 µs of these before and within 1.3 µs after (165 → 150, 200 → 187
  and 219 → 208 for se_e2_a), and its float64 numbers show the scan alone.

**What gave nothing.** Each on a probe build, alternated with the build as
it was, the least of four or five rounds, se_e2_a with tables, µs for the
88-atom frame in float32 unless said otherwise:

- The reads of a loop from global memory issued in groups of four or eight
  ahead of the arithmetic that waits for them, the sums in the order they
  had (the middle kernel of the fitting nets, the sums of the descriptor
  kernel, the first step of the force kernel): 167.3 → 168.9 and 167.6; at
  224 atoms 222.1 → 223.9 and 220.4. The same for the copies of the tile
  kernels into shared memory: 167.3 → 167.3, at 224 atoms 222.1 → 222.9.
- Two or four neighbor slots of a warp at a time in the force kernel, so
  that their reads overlap: 167.3 → 168.1 and 164.0 in float32, and worse in
  float64: 400 → 428 and 415 at 88 atoms, 731 → 784 and 765 at 224.
- A first read of the lines of the search input by all lanes at once before
  the scan: with the new scan 158.4 → 157.9.
- Blocks of 256 threads for the descriptor kernel: 165.9 → 164.7. Blocks of
  512 for the force kernel: 159.7 against 159.5 with 256, and at 224 atoms
  229.6 against 218.1 with 128.

**What this costs.**

- A profile is of an evaluation with marks in it: 180 µs for the 88-atom
  frame in float32 where a plain one takes 151, with up to 5 µs too many on
  a stage of so short a pass. Its numbers are not those of profiles printed
  before: `stages_ms` has `issue` and `device` where it had one number for a
  stage, and `bench_omnimd_gpu.py` writes it down as it comes.
- The force kernel has two widths, chosen by two numbers measured on one
  card (an RTX 4090 with 144 multiprocessors) with the HAP models and one
  water model; the evaluator's `--wide-forces` moves the limit.
- Passes of more than 224 centers, and float64 passes of more than 108, gain
  from the scan only.
- The forces of a frame evaluated alone and in a batch past the limit now
  differ by the rounding of the two widths (8e-14 eV/Å at most on the HAP
  frames) on top of what two runs differ by anyway (3e-14).

**Not done.** What an evaluation of the 88-atom frame still takes: the
fitting nets 62 of 158 µs, the search 28, the forces 25, the copy back with
the host's return 15, the descriptor 11, the tables 11 and the environment 6
(the device's times with one stage marked). The fitting nets were not made
shorter by reading their weights in groups (above); what they wait for was
not found. The search is one warp's work for a center from end to end, scan
and ranking; several warps for a center were not tried. A neighbor list kept
over several steps would take the search out of most evaluations of a run;
the engine has none. The widths were not tried between 108 and 162 centers
in float64 or between 224 and 264 in float32, nor with fewer than 54. All
times are of the evaluator's `--bench`: no run of the engine was timed. The
trainer's stage times (`training.profile_stages`) still wait for the device
after every stage.

**Tests** (on this machine; the GPU suites do not run in CI).
`neighbor_scan_takes_the_cells_of_any_grid_in_runs` (`gpu/tests/device.rs`)
has the search against the reference for periodic boxes with one, two and
five cells along the last axis, with one cell along the middle and along the
first, and a skewed one, and for an open slab and an open rod, each by the
warp path, by the serial path and with a buffer of seven;
`systems_in_one_search_do_not_see_each_other` has two open clusters stored
side by side. `marks_time_the_device_work_between_them` has the marks on a
stream of its own. In `native/tests/gpu_parity.rs`,
`few_centers_take_wider_blocks_in_the_two_body_force_kernels` has the width
a pass takes at the limit and one center past it, for frames, batches,
chunks and a committee, with the limit of each precision and for both
descriptors, with the same energies and forces within 1e-11 either way, and
`the_stages_of_an_evaluation_are_timed_by_the_device` has both clocks adding
up, for a pass in chunks, for a frame that leaves its tables and for an
evaluation that fails.

A mutation round on the scan, the marks, the stage clock, the sums of the
stage times and the choice of the block width had 85 mutants. 83 were caught
by the tests as first written, and two showed gaps that have checks now: an
open box that scans a cell too many (in the layout of a single system the
cells past the end read as empty; two open clusters stored side by side
catch it), and a setter that kept the limit at the default or below (a pass
of four frames with a limit above the default). Five more came with the
limit of float64 and were caught, and the eleven of the block width ran
again on the changed file: all 90 are caught.

Locally: workspace 234, engine with `native-gpu` 93, GPU crates 190, each in
release and debug; `cargo fmt --check` clean; the legacy `cuda` features
compile.

Files: `hap-eval-small-pass-rtx4090-2026-11-19.json` (before and after; and
`-r1`, the first run), `hap-eval-force-blocks-rtx4090-2026-11-19.json` (the
two widths), `hap-eval-stages-rtx4090-2026-11-19.json` (the stage times of
all three frames in both precisions),
`hap-eval-stages-one-marked-rtx4090-2026-11-19.json` (the probe with one
stage marked, beside a mark after every stage in the same session) and
`neighbor-search-scan-rtx4090-2026-11-19.json` (the search alone). The
probe builds are not in the repository, and neither are the numbers of the
paragraphs that name one without a record; the probes
`device_times_by_marks`, `arithmetic_costs` and `neighbor_search_costs` are
in `gpu/tests/costs.rs`.

# The rows of se_e3 laid out on the device, 2026-11-18

An se_e3 descriptor has a row for every pair of an atom's neighbors, and
the rows of a frame are laid out from the number of neighbors every atom
holds. The host made that layout: a kernel counted, the counts came back,
the host added up n(n + 1)/2 for every atom and sent the first rows down
again. That is a copy back to the host in the middle of every evaluation,
the one thing the pipeline otherwise avoids: the host waits for all that
is queued, and the device then stands still while the host queues the
rest. The two sections below name it as not done.

**What the host needs the rows for.** Three things: to launch the embedding
nets over them, to end a chunk where they would be too many, and to size
the buffers that hold them. With tables for the nets (since this morning)
nothing is launched over the rows: every kernel of the descriptor has a
block for an atom and reads the first row of its atom from the device.
What is left are the chunks and the sizes, and a frame that is one chunk
even if every neighbor slot of every atom is full needs neither answered:
its buffers are made for that many rows (an allocation from the pool costs
0.4 µs at any size from 8 KB to 256 MiB, `gpu/tests/costs.rs`), and the
rows of one model of a committee are that far from those of the next.

**What is done.** For such a pass a kernel of one block makes the layout
(`se_e3_layout`): a thread for every atom finds the number of its
neighbors of each type by halving (they fill the first slots of their
type), and a scan over the block's shared memory adds up the rows. Nothing
comes back from the device before the results. The conditions, all of
them: tables for every se_e3 descriptor of the model in this pass; at most
1,024 atoms in the pass, the threads of a block (`set_layout_atoms` takes
fewer); one chunk by atoms; and for every se_e3 descriptor atoms ×
slots(slots + 1)/2 rows within what a chunk may hold. Any other pass goes
as before, and `GpuPotential::count_downloads` counts those: a model
without tables, a frame that left its tables and is evaluated again with
the nets, a large frame, a large batch. New in the interface of
`native::gpu_model`: `LAYOUT_ATOMS`, `layout_atoms`, `set_layout_atoms`,
`count_downloads`, and `device_row_starts`, which brings the layout of a
table of slots back for the tests of the kernel; the evaluator's `--bench`
lines have `count_downloads`.

**Measured.** The build before (ab943ea) and this one with `--compress
0.5`, alternated in six rounds in one session, the least of each kept,
before → after. Trained HAP models on frames of the sealed test,
microseconds for energy, forces and virial (`--bench 3000`):

| frame | precision | se_e3 to 4 Å | se_e2_a to 6 Å with se_e3 to 4 Å | the earlier hybrid | se_e2_a to 6 Å, no se_e3 |
|---|---|---:|---:|---:|---:|
| 88 atoms | float32 | 154 → 135 | 245 → 225 | 176 → 151 | 163 → 164 |
| 88 atoms | float64 | 231 → 214 | 515 → 493 | 228 → 202 | 391 → 391 |
| 169 atoms | float32 | 192 → 172 | 302 → 282 | 187 → 165 | 199 → 200 |
| 169 atoms | float64 | 351 → 330 | 869 → 846 | 261 → 236 | 687 → 685 |
| 224 atoms | float32 | 217 → 196 | 319 → 299 | 202 → 181 | 218 → 217 |
| 224 atoms | float64 | 396 → 375 | 922 → 899 | 294 → 274 | 716 → 716 |

The water benchmark models (untrained), a frame of 192 atoms and one of
1,536, which is past the limit and goes as before:

| frame | precision | se_e3 to 6 Å | se_e2_a to 6 Å with se_e3 to 4 Å |
|---|---|---:|---:|
| 192 atoms | float32 | 446 → 422 | 313 → 291 |
| 192 atoms | float64 | 1,106 → 1,082 | 884 → 862 |
| 1,536 atoms | float32 | 2,212 → 2,211 | 1,239 → 1,238 |
| 1,536 atoms | float64 | 6,269 → 6,265 | 5,089 → 5,086 |

Batches of frames of 192 water atoms in one pass (`--bench-batches`),
microseconds a pass; six frames are past the limit:

| frames | atoms | precision | se_e3 to 6 Å | se_e2_a to 6 Å with se_e3 to 4 Å |
|---:|---:|---|---:|---:|
| 1 | 192 | float32 | 450 → 420 | 317 → 293 |
| 1 | 192 | float64 | 1,108 → 1,079 | 883 → 861 |
| 2 | 384 | float32 | 642 → 617 | 434 → 408 |
| 2 | 384 | float64 | 1,883 → 1,854 | 1,374 → 1,346 |
| 3 | 576 | float32 | 961 → 938 | 610 → 585 |
| 3 | 576 | float64 | 2,802 → 2,784 | 2,085 → 2,055 |
| 4 | 768 | float32 | 1,313 → 1,295 | 714 → 696 |
| 4 | 768 | float64 | 3,415 → 3,398 | 2,578 → 2,551 |
| 5 | 960 | float32 | 1,584 → 1,548 | 873 → 856 |
| 5 | 960 | float64 | 4,185 → 4,173 | 3,309 → 3,279 |
| 6 | 1,152 | float32 | 1,794 → 1,800 | 1,012 → 1,020 |
| 6 | 1,152 | float64 | 5,009 → 5,039 | 3,881 → 3,871 |

- With an se_e3 descriptor and the rows laid out on the device (42 cases)
  an evaluation is 12 to 36 µs shorter, 22 in the middle. For the HAP
  frames that is 6% to 14% of an evaluation in float32 (18 to 25 µs of 154
  to 319) and 2% to 12% in float64 (17 to 27 µs of 228 to 922).
- The model without se_e3 is the control: nothing on its path changed, and
  it is within 0.5% (-1.0 to +1.6 µs). The passes of more than 1,024 atoms
  go as before and are within 0.7%, which at their length is -30 to +10
  µs: what a session scatters at a millisecond and more.
- In batches the gain is the same up to 384 atoms (23 to 30 µs). From 576
  to 960 atoms the numbers say 12 to 36 µs, which is within that scatter
  for passes of 0.6 to 4 ms: such a pass waits for the device, not for the
  host, and the layout kernel runs on one multiprocessor.
- What it costs is device memory. The buffers of the rows are made for
  full slots, and the pool keeps them between evaluations: for the HAP
  se_e3 descriptor (76 slots, 2,926 rows an atom at most, 240 to 280 in
  these frames) 4.1 MB for 88 atoms where the rows take 0.34 and 10.5 MB
  for 224 where they take 1.0, in float32; for 960 water atoms (138 slots,
  9,591 rows an atom at most, 3,559 held) 147 MB where they take 55, and
  258 MB in float64. A pass whose full slots would not fit a chunk is laid
  out on the host as before.
- The batches are one frame of 192 atoms repeated, and all times are of
  the evaluator's `--bench`: no run of the engine was timed.
- A first run of four rounds (`...-r1.json`) was made on a machine less
  quiet: its control cases are up to 2.1% apart, and its gains on the HAP
  frames 15 to 35 µs. The tables are from the second run, of six rounds.
- Before the change a probe that left out the copy back alone (the counts
  of the first evaluation used again, the count kernel and the upload
  kept) had saved 18 to 30 µs on the HAP frames of 88 and 224 atoms.

**Tests** (`native/tests/gpu_parity.rs`, on this machine; the GPU suites do
not run in CI). The kernel against the sums it stands for, to the row: for
1, 2, 31, 32, 33, 63, 64, 65, 500, 1,023 and 1,024 atoms, with atoms that
hold nothing, atoms with every slot full, neighbors that are the atom 0, a
type without slots, the slots of the water model, and 6,000 slots, whose
rows pass thirty-two bits; 1,025 atoms are refused. Where the layout is
made, by the count of the passes that came back: with nets, with tables,
at the limit of atoms and one below it, in chunks one atom short of the
frame, with room for the rows of full slots and for ten rows less, in
batches of 1,000 and 1,125 atoms, for a frame that leaves the tables, in
float32, for a hybrid of two se_e3 descriptors and a radial one, and for
one whose se_e3 descriptors have five slots and 240; each time against the
CPU reference or against the layout made on the host.
The models of a committee, with every slot of every atom full, so that
the rows of one end where those of the next begin, and with a row budget
that holds one model's rows and not three.

A mutation round on the layout kernel, on what the host takes for the rows
and on when the layout is made on the device had 30 mutants. 27 were
caught by the tests as first written; three showed gaps that now have
checks:

- A bound one row an atom short of full slots. The test of the committee
  meant to have every slot of every atom full and did not: with two types
  the nearest neighbors of any type are taken first, as many as there are
  slots, and of those at most the slots of each type, which need not fill
  them. Nothing was at the bound, then. The test has one type now.
- The rows a chunk may hold, not shared among the models of a committee.
- Every se_e3 descriptor of a hybrid given the bound of the first.

With these all 30 are caught (the three, and the three next to them in the
list, ran again). The 42 mutants of the same file from the round of this
morning (the tables of a descriptor, the kernels that read them, the host
side) ran again on the changed file: all caught.

Locally: workspace 232, engine with `native-gpu` 93, GPU crates 186, each
in release and debug; `cargo fmt --check` clean; the legacy `cuda` features
compile.

**Not done.** The nets path keeps the copy back: its embedding kernel is
launched over the rows. So does the trainer. A pass of more than 1,024
atoms has it too, where it is 20 µs of milliseconds. The layout kernel
runs on one multiprocessor, with two barriers for every round of its scan;
for passes near its limit with thousands of pairs an atom the gain is
within what the timings scatter, and whether a faster scan would show
there was not tried.

Files: `hap-eval-se-e3-layout-rtx4090-2026-11-18.json` (and `-r1`, the
first run); the probe of allocations by size is `allocation_costs_by_size`
in `gpu/tests/costs.rs`.

# Tables in pieces of uneven length for se_e2_a and se_e2_r, 2026-11-18

`--compress` (`compress_min_distance` in the engine) now gives the nets of
se_e2_a and se_e2_r the kind of table that the nets of se_e3 got earlier
the same day (the section below): pieces as long as the net allows, each
within a tolerance of its net, found from the bits of the input. The tables
with a knot every `step` that these descriptors had (se_e2_a since
2026-11-10, se_e2_r since 2026-11-12) are gone, and their code with them
(`ScalarTable` of the gpu crate). What the tables are and how they are
made is in [`native/README.md`](../../native/README.md#embedding-tables).

**What changes for whoever asks for tables.**

- `table_step` (`:STEP` of `--compress`) is no distance between knots any
  more. It names a tolerance by the table of one step it stands for: 1e-13
  of the largest value of a net at 0.01 and (step/0.01)⁶ times that at
  another step, with 1e-14 at best in float64 and 1.5e-8 in float32.
- The tables reach to a fifth of the distance asked for at that tolerance
  all the way. They had ten times the step below the distance asked for.
- The tables of se_e3 end there too, exactly. They went on to the next
  power of two of θ, so that a pair somewhat closer than a fifth of the
  distance asked for was still looked up; such a frame now goes to the
  nets.
- A request for tables can now be refused for an se_e2_a or se_e2_r model
  as well, when a net turns faster than the precision of its input can
  follow; the potential then keeps its nets. None of the models here is.
- `omnimd-native-gpu-eval` prints the pieces of all tables.

**Against the nets.** The 83 frames of the HAP sealed test (largest force
component 6.2 eV/Å), `--compress 0.5`, four models of each kind, float64.
The largest difference from the same model with its exact nets, with the
tables of one step (the build of 67e730a) → with these; the tables of a
model, and the time to make them:

| models | energy (eV/atom) | force (eV/Å) | virial (eV) | tables (MiB) | made in (s) |
|---|---:|---:|---:|---:|---:|
| se_e2_a to 6 Å, the pipeline's models of iteration 4 (DeePMD-kit) | 8.4e-10 → 1.3e-13 | 3.9e-6 → 7.8e-11 | 2.0e-5 → 3.5e-10 | 392 to 393 → 202 to 216 | 0.28 to 0.31 → 0.49 to 0.56 |
| se_e2_a to 6 Å, trained by omniMD | 2.3e-11 → 1.3e-13 | 2.1e-7 → 1.4e-10 | 8.1e-7 → 6.8e-10 | 392 → 159 to 182 | 0.28 to 0.31 → 0.41 to 0.48 |
| se_e2_a to 4 Å, trained by omniMD | 2.1e-6 → 9.1e-14 | 1.1e-3 → 1.1e-10 | 6.4e-3 → 2.7e-10 | 352 → 135 to 148 | 0.25 to 0.26 → 0.37 to 0.42 |
| se_e2_r to 6 Å, trained by DeePMD-kit for 20,000 steps | 1.3e-13 → 1.8e-13 | 1.1e-10 → 1.1e-10 | 3.0e-10 → 4.4e-10 | 392 to 394 → 121 to 137 | 0.28 to 0.29 → 0.28 to 0.31 |
| se_e2_r to 6 Å, trained by omniMD | 7.1e-9 → 1.3e-13 | 9.0e-6 → 4.4e-11 | 4.9e-5 → 1.8e-10 | 392 → 165 to 183 | 0.28 to 0.30 → 0.40 to 0.44 |
| se_e2_a to 4 Å with se_e2_r to 6 Å, trained by omniMD | 2.1e-9 → 8.1e-14 | 5.5e-6 → 1.5e-10 | 1.3e-5 → 2.7e-10 | 744 → 306 to 323 | 0.52 to 0.55 → 0.78 to 0.85 |
| se_e2_a to 6 Å with se_e3 to 4 Å | 3.2e-12 → 1.3e-13 | 2.1e-8 → 8.6e-11 | 6.3e-8 → 3.6e-10 | 432 to 435 → 168 to 192 | 0.37 to 0.39 → 0.42 to 0.49 |
| the earlier hybrid, se_e2_a to 5 Å with se_e3 to 3.2 Å | 1.5e-11 → 1.3e-13 | 1.9e-7 → 2.1e-10 | 9.3e-7 → 4.3e-10 | 146 to 148 → 104 to 113 | 0.15 to 0.16 → 0.24 to 0.27 |
| se_e3 to 4 Å | 1.8e-13 → 1.8e-13 | 1.6e-10 → 1.6e-10 | 6.4e-10 → 6.4e-10 | 48 to 50 → 48 to 50 | 0.11 to 0.12 → 0.11 |

- With these tables the forces of every kind of model are within 2.1e-10
  eV/Å of those of its nets. The tables of one step were as close for one
  kind, the se_e2_r models that DeePMD-kit trained for 20,000 steps, whose
  weights are no larger than 1.1; for the others, whose weights reach 2.1
  to 6.0 (300,000 steps of omniMD, or the pipeline's own training), they
  were 2.1e-8 to 1.1e-3 away.
- The se_e2_a models with a cutoff of 4 Å (omniMD, 300,000 steps, the
  angular part of the hybrids of 2026-11-16 trained alone) are the kind the
  tables of one step missed most: 1.1e-3 eV/Å, in float32 as well (below).
  At the worst place of the worst net of one of them a table with a knot
  every 0.01 is 7e-4 of the largest value of the net off, and 1e-2 of its
  largest slope.
- What it costs in float64: the tables take longer to make for every kind
  but the one above, 0.24 to 0.85 s for a model where they took 0.15 to
  0.55 (the pipeline's models 0.5 s from 0.3). They are 1.3 to 3.2 times
  smaller (the pipeline's 1.8 to 1.9 times), not the factor of float32.
- se_e3 is as it was: the same pieces, to the same numbers.
- No frame left the tables, of either kind.
- (The section below has 2.2e-7 where the second row has 2.1e-7 for the
  tables of one step: the same 2.149e-7, rounded twice there.)

In float32 a table cannot be closer to its net than the net is computed,
so the differences are those of rounding. The largest difference of a force
component (eV/Å) from the float32 nets of the same model, and from its
float64 nets, where the float32 nets themselves are in the third column:

| models | from the float32 nets: tables of one step, these | float32 nets from the float64 nets | tables of one step from the float64 nets | these from the float64 nets | tables (MiB) | pieces |
|---|---:|---:|---:|---:|---:|---:|
| se_e2_a to 6 Å, the pipeline's models of iteration 4 (DeePMD-kit) | 6.4e-6, 9.8e-6 | 8.9e-6 | 8.7e-6 | 6.3e-6 | 196 to 197 → 13.3 to 14.3 | 5,659 to 6,103 |
| se_e2_a to 6 Å, trained by omniMD | 7.6e-6, 9.1e-6 | 7.7e-6 | 8.9e-6 | 7.3e-6 | 196 → 10.4 to 12.0 | 4,427 to 5,103 |
| se_e2_a to 4 Å, trained by omniMD | 1.1e-3, 5.4e-6 | 7.3e-6 | 1.1e-3 | 5.7e-6 | 176 → 8.9 to 9.6 | 3,785 to 4,060 |
| se_e2_r to 6 Å, trained by DeePMD-kit for 20,000 steps | 6.2e-6, 7.5e-6 | 5.8e-6 | 6.2e-6 | 4.4e-6 | 196 to 197 → 7.8 to 8.7 | 3,320 to 3,691 |
| se_e2_r to 6 Å, trained by omniMD | 1.9e-5, 1.8e-5 | 3.3e-5 | 3.1e-5 | 3.1e-5 | 196 → 10.7 to 11.9 | 4,553 to 5,061 |
| se_e2_a to 4 Å with se_e2_r to 6 Å, trained by omniMD | 6.6e-6, 6.4e-6 | 7.1e-6 | 8.0e-6 | 5.6e-6 | 372 → 20 to 22 | 8,569 to 9,180 |
| se_e2_a to 6 Å with se_e3 to 4 Å | 5.3e-6, 5.8e-6 | 5.2e-6 | 8.0e-6 | 6.3e-6 | 200 → 11.8 to 13.1 | 6,918 to 7,604 |
| the earlier hybrid, se_e2_a to 5 Å with se_e3 to 3.2 Å | 8.7e-6, 1.0e-5 | 6.6e-6 | 9.7e-6 | 7.7e-6 | 62 to 63 → 7.3 to 7.8 | 6,126 to 6,358 |
| se_e3 to 4 Å | 1.3e-5, 1.3e-5 | 8.5e-6 | 9.5e-6 | 9.5e-6 | 4.0 to 4.1 → 4.0 to 4.1 | 3,982 to 4,121 |

- From the float32 nets of their model these tables are further than the
  tables of one step were in five kinds (the first, the second, the fourth
  and the two hybrids with se_e3, by 9% to 53%), closer in three (the 4 Å
  models by a factor of 200, the other two by 1% and 4%), and the same for
  se_e3.
- From the float64 nets they are closer than the tables of one step were in
  all eight kinds with a two-body descriptor (by 2% for the se_e2_r models
  trained by omniMD). They are closer than the float32 nets themselves in
  six of them; for the two hybrids with se_e3 they are further (6.3e-6
  against 5.2e-6, 7.7e-6 against 6.6e-6), as for se_e3 alone.
- The float32 tables of a model with two-body descriptors are 8 to 25 times
  smaller than they were (the pipeline's 14 to 15 times: 13 to 14 MiB from
  196), and are made in 0.04 to 0.11 s where they took 0.07 to 0.38.

**The step.** The pipeline's model 0 on the same frames, by the step asked
for (float64; the float32 tables are the same 13.3 MiB and 5,659 pieces
from 0.005 to 0.05, all at the 1.5e-8 that float32 has at best):

| `table_step` | tolerance | force (eV/Å) | tables |
|---:|---:|---:|---|
| 0.005 | 1e-14 (the best) | 1.3e-11 | 58,377 pieces, 268 MiB |
| 0.01 | 1e-13 | 7.8e-11 | 44,096 pieces, 202 MiB |
| 0.02 | 6.4e-12 | 2.5e-9 | 20,915 pieces, 96 MiB |
| 0.05 | 1.6e-9 | 2.7e-7 | 8,027 pieces, 37 MiB |

With the step 0.05 the tables are a tenth of what the tables of one step
took at 0.01 (393 MiB), and nine times closer than those were on this
model (2.4e-6 eV/Å).

**Every model on disk.** All 150 `*.native.json` under `bench/cases`, one
frame each (the 88 atoms of bulk HAP for the models trained on the HAP
data, the 192 atoms of the water benchmark for its five models),
`--compress 0.5`, both builds, both precisions. Three of the files are
empty and are no models (`hap-iter0-hybrid/replay-4k/part-0`, `part-1`,
`hap-iter0-pre/omnimd-float32`); of the 147 models none is refused by
either build in either precision, and no frame leaves the tables.

| | tables of one step | these |
|---|---:|---:|
| float64: force from the nets (eV/Å), median of the models | 6.4e-10 | 1.4e-11 |
| largest | 8.9e-4 | 7.5e-11 |
| models further than 1e-10 | 79 | none |
| further than 1e-7 | 18 | none |
| further than 1e-6 | 6 | none |
| float64 tables (MiB), median and most | 392, 748 | 142, 350 |
| time to make (s), median and most | 0.29, 1.08 | 0.39, 0.86 |
| float32: force from the float32 nets (eV/Å), median | 2.9e-6 | 3.1e-6 |
| largest | 8.9e-4 | 1.8e-5 |
| float32 tables (MiB), median and most | 196, 374 | 9.4, 23 |
| time to make (s), median and most | 0.20, 0.40 | 0.05, 0.11 |

**Where the tables of one step were closer.** A net as smooth as an
untrained one is followed to rounding by a table with a knot every 0.01,
and by these tables to their tolerance. The models of the water benchmark
(seeded weights, 0.64 at most), 192 atoms, float64, the largest difference
of a force component from the nets: se_e2_a 8.5e-16 → 1.2e-14 eV/Å, se_e2_r
2.0e-16 → 9.9e-15, the hybrid of the two 4.7e-16 → 1.9e-13 (its largest
force is 0.037 eV/Å; with `table_step` 0.005 it is 5.5e-15). Against
DeePMD-kit's uncompressed model in float64 that is what the tables are off
by, 1.2e-14, 9.9e-15 and 1.9e-13 eV/Å at 192 and 1,536 atoms, where the
records of 2026-11-12 and 2026-11-16 have 8.6e-16, 2.0e-16 and 5.0e-16 at
192.

**Against DeePMD-kit on the HAP frames.** The pipeline's four models and
DeePMD-kit 3.1.3 (`DeepPot` on the CPU, float64), all 83 frames, the
largest difference over the four models (`hap_check.py` with both builds
today; the build before gives the numbers of 2026-11-11):

| omniMD mode | energy per atom (eV) | force (eV/Å) | virial (eV) |
|---|---:|---:|---:|
| float64 | 1.8e-13 | 5.6e-13 | 1.1e-12 |
| float64, tables | 8.4e-10 → 1.8e-13 | 3.9e-6 → 7.8e-11 | 2.0e-5 → 3.5e-10 |
| float32 | 3.4e-7 | 8.9e-6 | 1.1e-4 |
| float32, tables | 4.4e-7 → 2.9e-7 | 8.7e-6 → 6.3e-6 | 8.1e-5 → 5.0e-5 |

The errors against the DFT labels move by 6.1e-13 at most with float64
tables (eV per atom or eV/Å; 1.1e-8 with the tables of one step).

**Speed.** An evaluation costs what it did. Both builds with `--compress
0.5` on the same cases, alternated in four rounds in one session, the least
of each kept, before → after. HAP frames of the sealed test, microseconds
for energy, forces and virial (`--bench 3000`):

| frame | precision | se_e2_a to 6 Å, trained by omniMD | the pipeline's model 0 | se_e2_r to 6 Å | se_e2_a to 6 Å with se_e3 to 4 Å |
|---|---|---:|---:|---:|---:|
| 88 atoms | float32 | 181 → 182 | 183 → 181 | 139 → 141 | 280 → 281 |
| 88 atoms | float64 | 447 → 444 | 446 → 445 | 220 → 216 | 600 → 591 |
| 169 atoms | float32 | 221 → 220 | 221 → 220 | 171 → 171 | 339 → 338 |
| 169 atoms | float64 | 764 → 758 | 768 → 762 | 324 → 322 | 969 → 964 |
| 224 atoms | float32 | 243 → 242 | 242 → 242 | 193 → 191 | 358 → 356 |
| 224 atoms | float64 | 803 → 798 | 802 → 788 | 356 → 347 | 1,019 → 1,017 |

The water benchmark models, ms:

| atoms | precision | se_e2_a | se_e2_r | hybrid of se_e2_a and se_e2_r |
|---:|---|---:|---:|---:|
| 12,288 | float32 | 5.67 → 5.68 | 4.24 → 4.24 | 6.30 → 6.32 |
| 12,288 | float64 | 37.2 → 36.9 | 15.7 → 15.2 | 38.1 → 37.4 |
| 98,304 | float32 | 46.1 → 46.1 | 34.9 → 34.9 | 49.8 → 50.1 |
| 98,304 | float64 | 295 → 292 | 124 → 121 | 302 → 297 |

- After over before is 0.969 to 1.011 over the 36 cases: 0.989 to 1.011 in
  float32 (0.999 on average) and 0.969 to 0.997 in float64 (0.987). No
  slower on average in either precision, then; the largest difference of a
  case is 1.1% up, in float32. (The lookup through the cells of a table
  takes the place of a subtraction and a multiplication by the reciprocal
  of the step.)
- The tables of the water models (untrained, 4 nets, or 8 for the hybrid)
  are 6.2, 6.4 and 13 MiB in float64 where they were 17, 17 and 34, and
  0.4, 0.4 and 0.9 MiB in float32 where they were 8.6, 8.6 and 17.
- `bench_omnimd_gpu.py` on the three water cases (one run each, in the
  files below) gives the times with nets and with tables at all four
  sizes: se_e2_a with tables 0.256, 0.959, 5.73 and 46.2 ms in float32 and
  0.791, 4.79, 36.8 and 295 in float64.

**Tests.** `native/src/adaptive_table.rs` (run in CI): a table that counts
from elsewhere than zero and has one side, against its net, with the cells
of the side it lacks naming the pieces of the other; a feature that lies
between all the points a piece is checked at when the units are followed
over the places of the piece and not over its inputs; an origin that is no
number. `gpu/tests/device.rs`: a table that reaches less far than its
pieces go and not at all below zero, with inputs within, at the reach, past
it, past the pieces, below zero and not a number, each looked up within the
table and reported as the reach says, to the bit, in both precisions; what
is no reach, refused. `native/tests/gpu_parity.rs`: a pair of atoms just
outside and just inside a fifth of the distance asked for, for centers of
either type and for se_e3, goes to the tables and to the nets; models whose
nets do not depend on their input have one piece a table and side, which
counts the tables of every kind of descriptor and their sides (one side for
se_e2_a and se_e2_r, two for se_e3, one net for the two orders of a pair
of neighbor types, the members of a committee each their own), in both
precisions, with the size that goes with it; a net of an se_e2_a model
that float32 cannot tabulate, refused with its model, its types and its
place, while the committee goes on with its nets, and the same for se_e3
in a committee; a neighbor just inside the cutoff, down to the last
distance below it that a float64 holds, for several cutoffs and smoothing
radii, centers of either type and both precisions, stays in its table (the
switch there is rounding, and a place below zero would send the frame to
the nets; [`switch_rounding.py`](../../native/tools/switch_rounding.py)
shows by exact arithmetic that the kernel as written gives none, and for
which radii a product fused into its subtraction would).

A mutation round on the builder (34 mutants), the lookup and the reach on
the device (19), and the tables of a descriptor, the environment kernel,
the kernels that read the tables of se_e3 and the host side (42): all 95
are caught. Three of the checks above were written with the list and are
the only ones that catch their mutants: pieces on a side that no input
reaches (a table left right and only larger), the interval the units are
followed over for a table with an origin, and the model and types in the
message of a refusal. One mutant is left out as equivalent: pieces of a
two-body table to the power of two beyond the next one, where no input is
looked up. The first 76 ran before the last commit of the code, which
changed comments and usage texts; the 34 of the builder ran again after
the last change of its tests (the comparison of a table of one side with
the table of two, within rounding where it was bit for bit) and are caught
as before.

Locally: workspace 232, engine with `native-gpu` 93, GPU crates 183, each
in release and debug; `cargo fmt --check` clean; the legacy `cuda` features
compile. The GPU suites do not run in CI.

**Removed.** `ScalarTable`, `TableSegments`, `quintic_coefficients` and the
`scalar_table` kernel of the gpu crate, with their tests: nothing used them
any more, and an evaluation with the tables that replace them is within
1.1% of what it was in each of the 36 cases above, and below it on average.

**Not built.** Tables for the CPU reference `NativePotential` and for the
trainer. A tolerance below 1e-14 in float64, which a net as smooth as an
untrained one would allow. The two things the section below leaves for
se_e3: the small download of the neighbor counts in every evaluation, and
going through the pairs of an atom by piece.

Files: `hap-tables-against-nets-rtx4090-2026-11-18.json`,
`tables-of-every-model-rtx4090-2026-11-18.json`,
`hap-eval-tables-before-and-after-rtx4090-2026-11-18.json`,
`hap-check-tables-rtx4090-2026-11-18.json` (and
`hap-check-tables-of-one-step-rtx4090-2026-11-18.json`, the build before),
`omnimd-gpu-tables-rtx4090-2026-11-18.csv`,
`omnimd-gpu-se-r-tables-rtx4090-2026-11-18.csv`,
`omnimd-gpu-hybrid-tables-rtx4090-2026-11-18.csv`.

# Tables for the embedding nets of se_e3, 2026-11-18

`--compress` (`compress_min_distance` in the engine) now gives the nets of
an se_e3 descriptor tables too, alone or in a hybrid, where they kept their
networks before. They are not DeePMD-kit's kind of table: their pieces are
of uneven length, each within a tolerance of its net, and the kernels read
them where they read rows of embedding values before. What they are and
how they are made is in
[`native/README.md`](../../native/README.md#embedding-tables).

**Why not a table of one step.** These nets take θ, the product of the
normalized directions of two neighbors. For the HAP model trained on
2026-11-17 (se_e3 to 4 Å) θ spans ±1,400 for neighbors from 0.9 Å, and the
first layers of its nets have weights up to 4.5: a net turns within a
hundredth of a unit of θ where its data is. Quintic tables with a knot
every step of θ over that range, against the nets at 20,000 random θ, the
worst of the fifteen nets
([`se_e3_uniform_tables.py`](../se_e3_uniform_tables.py)):

| step | value off by (of the largest value) | slope off by (of the largest slope) | all nets, float32 |
|---:|---:|---:|---:|
| 0.1 (DeePMD-kit's for this descriptor) | 2.4e-2 | 2.7e-1 | 404 MB |
| 0.05 | 2.2e-3 | 9.1e-2 | 808 MB |
| 0.02 | 3.5e-4 | 1.2e-2 | 2.0 GB |
| 0.01 | 9.8e-6 | 7.3e-5 | 4.0 GB |

The tables in pieces of uneven length for the same nets: 4,092 pieces in
4.1 MiB in float32 (every piece within 1.5e-8 of the largest value of its
net at the three places it is checked at) and 26,966 pieces in 50 MiB in
float64 (within 1e-13 there), made in 0.1 s.

**Against the nets.** The 83 frames of the HAP sealed test (largest force
component 6.1 eV/Å), `--compress 0.5`, four models of each kind, the
largest difference from the same model with its exact nets:

| models | precision | energy (eV/atom) | force (eV/Å) | virial (eV) | tables |
|---|---|---:|---:|---:|---|
| se_e3 to 4 Å | float64 | 1.8e-13 | 1.6e-10 | 6.4e-10 | 26,966 pieces, 50 MiB |
| se_e3 to 4 Å | float32 | 2.1e-7 | 1.4e-5 | 1.4e-4 | 4,092 pieces, 4.1 MiB |
| se_e2_a to 6 Å with se_e3 to 4 Å | float64 | 3.2e-12 | 2.1e-8 | 6.3e-8 | 433 MiB, 40 of them se_e3's |
| the earlier hybrid | float64 | 1.5e-11 | 1.9e-7 | 9.3e-7 | 147 MiB |
| se_e2_a to 6 Å, trained by omniMD | float64 | 2.3e-11 | 2.2e-7 | 8.1e-7 | 392 MiB |
| se_e2_a to 6 Å, the pipeline's models of iteration 4 (DeePMD-kit) | float64 | 8.4e-10 | 3.9e-6 | 2.0e-5 | 393 MiB |

- In float32 the tables are no further from the float64 nets than the
  float32 nets are: 9.5e-6 eV/Å against 8.5e-6 for se_e3 alone, and 8.0e-6
  to 9.7e-6 against 5.2e-6 to 8.9e-6 for the other four kinds.
- The last two rows have no se_e3: they are the tables of one step that
  se_e2_a has had all along, measured here for the first time on trained
  models. They follow a smooth net to rounding level (the untrained water
  models below) and a trained one to 1e-7 or 1e-6 of a force; half the step
  takes one of the pipeline's models from 2.4e-6 to 1.4e-7 eV/Å and a
  quarter to 2.8e-9. In a hybrid they are what the difference from the
  nets comes from. Tables in pieces of uneven length for these nets as
  well would be smaller and closer; that was not built.
- No frame left the tables. A frame that does is evaluated with the nets
  (the shortest distance in these frames is 0.890 Å; the tables of se_e3
  reach to a fifth of the 0.5 Å asked for).

**Speed on the HAP frames.** The fully trained models of 2026-11-17 on
three frames of the sealed test, microseconds for energy, forces and
virial, the least of four alternated rounds of `--bench 3000`:

| frame | precision | embedding | se_e2_a to 6 Å | with se_e3 to 4 Å | the earlier hybrid | se_e3 to 4 Å |
|---|---|---|---:|---:|---:|---:|
| 88 atoms | float32 | nets | 209 | 332 | 210 | 204 |
| 88 atoms | float32 | tables | 183 | 284 | 197 | 175 |
| 88 atoms | float64 | nets | 737 | 1,238 | 394 | 596 |
| 88 atoms | float64 | tables | 448 | 604 | 254 | 267 |
| 169 atoms | float32 | nets | 265 | 444 | 225 | 281 |
| 169 atoms | float32 | tables | 224 | 344 | 219 | 219 |
| 169 atoms | float64 | nets | 1,264 | 2,132 | 455 | 1,048 |
| 169 atoms | float64 | tables | 768 | 963 | 293 | 388 |
| 224 atoms | float32 | nets | 301 | 526 | 260 | 329 |
| 224 atoms | float32 | tables | 244 | 357 | 231 | 242 |
| 224 atoms | float64 | nets | 1,539 | 2,769 | 635 | 1,403 |
| 224 atoms | float64 | tables | 804 | 1,025 | 333 | 439 |

- The pipeline's descriptor with the three-body one beside it (the model
  whose force error is 8 to 9 percent lower) takes 0.28 to 0.36 ms in
  float32 with tables, 1.5 times the pipeline's own model with tables
  (0.18 to 0.24 ms), and 1.3 times in float64 (0.60 to 1.03 against 0.45
  to 0.80 ms). With tables for its se_e2_a descriptor alone, as the build
  before had them, the same model took 0.32, 0.41 and 0.48 ms in float32
  and 0.95, 1.67 and 2.06 in float64 in the same session: the tables of
  se_e3 take 10 to 27 percent off that in float32 and 37 to 50 percent in
  float64.
- This corrects the section below, which put the cost of the second
  descriptor at 1.6 to 2.0 times an evaluation and named the missing
  tables beside it. In float32 the two models were 0.13 to 0.24 ms apart
  with tables for se_e2_a alone and are 0.10 to 0.12 with tables for both:
  the nets of the three-body descriptor were a quarter of the difference at
  88 atoms and half at 224, and the rest is its other stages (the rows of
  its pairs, with a small download of the neighbor counts in every
  evaluation, its components, its two force kernels). In float64 the nets
  were most of it.
- se_e3 alone: 1.2 to 1.4 times faster in float32, 2.2 to 3.2 times in
  float64. All numbers of this table are from one session; the same build
  measured some 9% lower without tables on 2026-11-17 than it does today
  (the build before and this one are within 1% of each other on the nets
  today), so they are not to be set against that day's.

**Speed on the water benchmark** (untrained models, ms per evaluation, the
faster of two runs, `bench_omnimd_gpu.py`). se_e3 alone, to 6 Å with `sel`
[46, 92] and embedding [10, 20, 40], 3,600 pairs of neighbors an atom.
DeePMD-kit: LAMMPS with its exact nets, 2026-11-15; its own tables for this
descriptor do not reproduce its model (the section of 2026-11-16 below), so
there is no DeePMD-kit column for tables:

| atoms | precision | DeePMD-kit, nets | omniMD, nets | omniMD, tables | chunks |
|---:|---|---:|---:|---:|---|
| 192 | fp64 | 92.3 | 9.76 | 1.25 | 1 |
| 1,536 | fp64 | 6,494 | 73.9 | 7.11 | 1 |
| 12,288 | fp64 | out of memory | 590 | 53.2 | 5, with tables 1 |
| 98,304 | fp64 | error | 4,693 | 425 | 39, with tables 2 |
| 192 | fp32 | 28.8 | 2.05 | 0.485 | 1 |
| 1,536 | fp32 | 219 | 14.1 | 2.50 | 1 |
| 12,288 | fp32 | error | 110 | 19.1 | 3, with tables 1 |
| 98,304 | fp32 | error | 875 | 153 | 20, with tables 1 |

The hybrid of se_e2_a to 6 Å and se_e3 to 4 Å (`sel` [16, 32]), now with
tables for both descriptors (DeePMD-kit with the atoms held, 2026-11-16):

| atoms | precision | DeePMD-kit, nets | omniMD, nets | omniMD, tables |
|---:|---|---:|---:|---:|
| 192 | fp64 | 25.2 | 2.37 | 1.00 |
| 1,536 | fp64 | 142 | 16.4 | 5.89 |
| 12,288 | fp64 | no result | 129 | 44.0 |
| 98,304 | fp64 | error | 1,026 | 348 |
| 192 | fp32 | 14.1 | 0.506 | 0.350 |
| 1,536 | fp32 | 42.5 | 2.92 | 1.33 |
| 12,288 | fp32 | no result | 20.2 | 8.16 |
| 98,304 | fp32 | error | 161 | 67.8 |

- With tables the float64 result of se_e3 at 192 atoms is DeePMD-kit's
  uncompressed model (CPU, float64) within 9.5e-15 eV per atom, 6.8e-14
  eV/Å and 4.1e-13 eV of virial, the hybrid's within 2.8e-13 eV/Å at 192
  and 1,536 atoms; float32 with tables is 9.1e-9 eV/Å from it, as float32
  with nets is (9.2e-9). At the larger sizes the tables are within 6.8e-14
  eV/Å (se_e3) and 2.8e-13 (hybrid) of omniMD's float64 nets.
- se_e3 with tables is 4.2 to 5.8 times faster than with nets in float32
  and 7.8 to 11.1 times in float64; at 192 atoms in float32 59 times
  faster than DeePMD-kit's exact nets in LAMMPS, where most of a LAMMPS
  step does not grow with the atoms.
- A pair takes 16 bytes with tables in float32 where it took 336 with its
  embedding rows, so 98,304 atoms (350 million pairs) are one chunk
  instead of twenty.
- Where the time goes at 192 atoms (stage times, each with a
  synchronization). Float32, 2.12 ms with nets: embedding nets 1.12,
  forces 0.37, descriptor 0.25. With tables 0.49 ms: descriptor 0.14,
  forces 0.11, rows of the pairs 0.10, neighbor search 0.07, fitting 0.07.
  Float64, 9.72 ms with nets: embedding nets 8.08. With tables 1.24 ms:
  forces 0.47, descriptor 0.42.

**How the kernels are laid out.** A first version put the tables where the
nets were: a kernel wrote the values and slopes of every pair, and the
descriptor and force kernels read them as before (192 water atoms: 1.22 ms
in float32, 3.70 in float64). Reading the pieces in those kernels instead,
with no rows stored, was faster in float32 (0.81 ms) and slower in float64
(4.22): in double precision the time went with the number of instructions
the warps issued, whatever the number of lanes at work in one, and a warp
for a row has most lanes idle most of the time. With a thread for a row
the same work took 1.25 ms in float64 and 0.49 in float32. The components
keep a warp for a row in float32, which is faster for frames of a few
hundred pairs an atom. The measurements are with the kernels in
`native/src/gpu_model.rs` and the rule in
[`gpu/README.md`](../../gpu/README.md).

**Tests.** `native/src/adaptive_table.rs` and `native/src/network.rs`
(run in CI): tables against nets at 40,000 points each, spread over the
range and packed near zero, for nets that turn within a fraction of a unit
and over hundreds; a net whose feature lies between all points a piece is
checked at (two units of a second layer with a gain of a thousand: a
builder that compared values alone keeps one piece that is zero
everywhere); every activation, residual and time step; that pieces meet in
value and slope and that the cells name every piece and no other, with
cells that lie within a piece and cells that hold several; the lookup in
float32
throughout; a net too fast for float32; a net whose own rounding is above
the tolerance; the bound on that rounding against what cancelling terms
lose. `gpu/tests/device.rs`: the lookup on the device against a table laid
out by hand whose quintics say which piece was read, to the bit, in both
precisions, with inputs at the ends of pieces and cells, at zero of either
sign, denormal, at the last float inside and outside. `native/tests/
gpu_parity.rs`: se_e3 with tables against the CPU reference in chunks by
atoms and by rows, in batches, in committees whose members have ranges of
their own, for atoms without a pair, with embeddings of 40, 136 and 200
outputs, in float32; a pair of atoms just inside and just outside the end
of the tables, and a batch with one frame outside, which alone goes back to
the nets; a coarse step that is visibly worse; a net that float32 cannot
tabulate, refused with its place while the potential goes on with its
nets; hybrids with tables for every descriptor, each shown to be in use.
`engine/tests/se_e3.rs`: an NVE run with tables follows the CPU run within
1e-9 eV.

A mutation round on the builder, the bounds on movement and rounding, the
lookup, the range and the nets of a descriptor's tables, the kernels that
read them and the host side had 72 mutants. 69 were caught by the tests as
first written, and three showed gaps that now have checks: the range the
movement of the units is bounded over on the negative side of zero, that a
chunk holds more pairs with tables, and the allowance for the rounding of a
net. The third did more than add a check. Its test net had a weight of few
bits, whose products with the places the builder looks at are exact, so the
rounding never showed there; with a weight of all its bits the builder
halved down to the last bit of θ and failed, because what the values at the
ends of a piece are off by shows in its slope the more, the narrower the
piece is. The allowance now covers that, with a mutant of its own. After
that correction the 35 mutants of the builder and of the bounds were run
again on the final code (the other 38, of the lookup, the kernels and the
host side, are in files that did not change after their run). One of them
was no longer caught: it adds pieces that no cell names, which leaves a
table right and only larger, and it had been caught by a limit on the
number of pieces that the corrected builder no longer reaches. A check that
the cells of a table name every one of its pieces catches it now. Four gaps
in all, then, and all 73 mutants are caught. Locally: workspace 233, engine with `native-gpu` 93, GPU
crates 182, each in release and debug; `cargo fmt --check` clean; the
legacy `cuda` features compile. The GPU suites do not run in CI.

**A change for hybrids.** `--compress` gives a hybrid tables for every
descriptor or for none: when a net of an se_e3 descriptor cannot be
tabulated (it turns faster than the precision of θ can follow) the request
is refused and the potential keeps all its networks, where before this
change the two-body descriptors of such a hybrid would have had their
tables. None of the twelve trained models here (three kinds with se_e3,
four seeds) is refused, in float32 or in float64.

**Not built.** Tables in pieces of uneven length for the nets of se_e2_a
and se_e2_r (see above: theirs are eight times larger and a thousand times
less close on these models). The small download of the neighbor counts
that an se_e3 descriptor makes in every evaluation, which a frame of a
hundred atoms feels. Going through the pairs of an atom by piece, so that
the outputs of a piece are worked out once for all its pairs. The trainer
does not use tables.

Files: `hap-se-e3-uniform-tables-2026-11-18.json`,
`hap-se-e3-tables-against-nets-rtx4090-2026-11-18.json`,
`hap-eval-se-e3-tables-rtx4090-2026-11-18.json`,
`omnimd-gpu-se-e3-tables-rtx4090-2026-11-18.csv` (and `-r2`),
`omnimd-gpu-hybrid-e3-tables-rtx4090-2026-11-18.csv` (and `-r2`).

# Training se_e3, 2026-11-17

`omnimd-native-gpu-train` trains the three-body descriptor se_e3 (`se_at`,
`se_a_3be`), alone or in a hybrid: with this the trainer has every
descriptor a native model can have. A request writes it as a DeePMD input
script does. Its embedding nets take the pairs of every atom's neighbors,
one net for a pair of neighbor types in either order; the pairs, the
descriptor and the forces come from the kernels of the inference path, the
tangent and the reverse pass are the trainer's. Contract, checks and the
full tables are in [`native/GPU_TRAINING.md`](../../native/GPU_TRAINING.md).

All runs below: the 171 frames of the HAP pipeline's iteration-0 training
(54 to 224 atoms, five types), one frame per step, the pipeline's schedule
(learning rate 1e-3 to 3e-8, energy and force terms), scored on its 83
sealed-test frames. Three models:

- **se_e3 to 4 Å**: `sel` [16, 8, 20, 24, 8] (room for every neighbor of
  these frames), the embedding sizes DeePMD-kit's input has by default for
  this descriptor, [10, 20, 40], fitting nets [240, 240, 240]. Its frames
  have 28,000 pairs on average and 70,000 at most.
- **se_e2_a to 6 Å with se_e3 to 4 Å**: the pipeline's descriptor with the
  one above beside it.
- **The earlier hybrid**: the architecture of the HAP project's models of
  October 2026 (`results/t2-fit-diagnostic-20261014/model-architecture.json`
  there): se_e2_a to 5.0 Å with [16, 32, 64] and 8 axis neurons, se_e3 to
  3.2 Å with [8, 16, 32], fitting [64, 64], narrow smoothing ranges and
  slots that are short on these frames (in 13,000 and 2,200 of 99,370
  selections of an atom and a neighbor type). Only the architecture is
  theirs: that diagnostic fitted eight frames with LBFGS and is not
  reproduced here.

The pipeline marks its models as not production-qualified.

**The same step as DeePMD-kit.** omniMD was given DeePMD-kit's initial
model and the frames in the order DeePMD-kit took them in a run of 4,000
steps, in float64 ([`bench/deepmd_replay.py`](../deepmd_replay.py)):

| | nets of se_e2_a | nets of se_e3 | fitting nets | sealed test, both models |
|---|---:|---:|---:|---|
| se_e3 to 4 Å: moved by DeePMD-kit | | 1.8e-2 | 1.0e-2 | |
| se_e3 to 4 Å: omniMD away from DeePMD-kit | | 9.4e-11 | 6.7e-11 | 0.495201 eV/Å, 66.5594 meV/atom |
| the earlier hybrid: moved by DeePMD-kit | 1.6e-2 | 2.0e-2 | 1.6e-2 | |
| the earlier hybrid: omniMD away from DeePMD-kit | 3.4e-8 | 4.5e-8 | 3.3e-8 | 0.622925 eV/Å; 178.7625 and 178.7623 meV/atom |

(Mean absolute difference of the trained numbers; each net of se_e3
counted once.) Both follow DeePMD-kit's displayed errors to the three
digits it prints. The hybrid is three hundred times further off because of
the run, not the step: a change of DeePMD-kit's initial model by one part in
1e11 has grown 2.2 times at the end of the first run and 2,000 times at the
end of the second (the same for one part in 1e9), so both distances are
what a difference of 2e-11 to 4e-11 at the start would leave. Two replays
of each run end the same distance from DeePMD-kit in every printed digit.
This adds to what the replay of the hybrid of se_e2_a and se_e2_r (the
section below) had covered: the gradient through se_e3, alone and in a
hybrid, through se_e2_a with a net per pair of types, and lists and
descriptors that drop neighbors. The virial term was off here too.

**Speed.** One request, steps per second, the best of three alternated
rounds (20,000 steps in float32, 6,000 in float64); DeePMD-kit 3.1.3
(PyTorch backend, float64, 2,000 steps) on the same requests:

| model | parameters | omniMD float32 | omniMD float64 | DeePMD-kit float64 |
|---|---:|---:|---:|---:|
| se_e2_a to 6 Å, a net per pair of types | 2,664,455 | 535 | 245 | 10.4 (2026-11-13) |
| se_e3 to 4 Å | 647,405 | 1,293 | 355 | 19.3 |
| se_e2_a to 6 Å with se_e3 to 4 Å | 2,728,655 | 414 | 160 | 7.04 |
| the earlier hybrid | 273,605 | 1,486 | 647 | 7.75 |

- omniMD trains the three 18, 23 and 83 times faster than DeePMD-kit with
  float64 networks, and 67, 59 and 192 times with float32 ones.
- Four requests at once in float32 run at 406, 121 and 538 steps/s each.
- A float32 step of se_e3 alone takes 0.96 ms with every stage timed, the
  reverse pass of its embedding nets 0.21 of it. That its rows come in a
  run for every atom and pair of types costs nothing: with all atoms as
  one type, where the rows of the one net are a single run, that pass
  takes 0.210 ms against 0.211.
- The inference path changed with this (the rows of an atom's pairs go by
  the pair of neighbor types; the force kernel is two kernels) and is no
  slower: se_e3 alone 2.31 and 2.30 ms before and after at 192 atoms in
  float32, 124.3 and 124.4 at 12,288; the hybrid of se_e2_a with se_e3
  0.594 and 0.566, 24.0 and 22.8.

**The full schedule: is the three-body descriptor worth it on this data?**
300,000 steps, float32 networks, four seeds at once. Smallest and largest
of the four models over all 83 frames (se_e2_a to 6 Å: the run of
2026-11-10, the same requests without the three-body descriptor, by the
build of that day; se_e2_a to 4 Å: the morning of 2026-11-17; both scored
again with the others):

| model | parameters | force RMSE (eV/Å) | energy RMSE (meV/atom) |
|---|---:|---:|---:|
| se_e2_a to 6 Å (the pipeline's model) | 2,664,455 | 0.0698 to 0.0723 | 1.64 to 1.78 |
| se_e2_a to 6 Å with se_e3 to 4 Å | 2,728,655 | 0.0641 to 0.0659 | 1.52 to 1.71 |
| the earlier hybrid | 273,605 | 0.0933 to 0.0949 | 1.28 to 1.38 |
| se_e2_a to 4 Å alone | 2,664,455 | 0.0947 to 0.0961 | 1.88 to 2.11 |
| se_e3 to 4 Å alone | 647,405 | 0.1282 to 0.1352 | 2.00 to 2.46 |

- Beside the pipeline's se_e2_a the three-body descriptor takes 8 to 9
  percent off the force error: each of its four models is below each of
  the four without it, and by system its four lie below the others in five
  of the eight, lower with overlap in two, the same in one. The energies
  are a little lower overall, with ranges that overlap.
- It costs 23 percent of the training speed and 1.6 to 2.0 times the time
  of an evaluation (0.30 to 0.48 ms at 88 to 224 atoms in float32 against
  0.19 to 0.28; 0.29 to 0.44 against 0.17 to 0.22 with embedding tables,
  which se_e3 does not have). (It has them since 2026-11-18, and with
  tables for both the factor is 1.5 in float32 and 1.3 in float64; the
  section above also says what the difference is made of.)
- The earlier hybrid, with a tenth of the parameters, has the force error
  of se_e2_a to 4 Å alone and the lowest energy error of the five; it
  trains 2.8 times faster than the pipeline's model and evaluates as fast
  in float32. se_e3 to 4 Å alone is not enough for this data.
- DeePMD-kit was not run on the full schedule (4.3 and 11.8 hours a model
  at its measured rates). Four seeds each; nothing here is a statement
  about the physics.

Files: `hap-train-speed-se-e3-rtx4090-2026-11-17.json`,
`hap-train-se-e3-replay-deepmd-...`, `hap-train-ensemble-se-e3-float32-omnimd-...`,
`hap-eval-se-e3-models-...`. The full runs shared the card with the test
suites for most of their time, so their own times are not rates.

# Training hybrids, 2026-11-17

`omnimd-native-gpu-train` trains a hybrid of se_e2_a and se_e2_r
descriptors: each descriptor of the list with its own cutoffs, `sel`, nets
and statistics, the fitting nets on the components of all of them, one
neighbor list for all as in the inference path. A request writes it as a
DeePMD input script does (`{"type": "hybrid", "list": [...]}`). (A model
with an se_e3 descriptor had no trainer yet that morning; see above.)
Contract, checks and the full tables are in
[`native/GPU_TRAINING.md`](../../native/GPU_TRAINING.md).

All runs below: the 171 frames of the HAP pipeline's iteration-0 training
(54 to 224 atoms, five types), one frame per step, the pipeline's networks
(embedding [25, 50, 100], 16 axis neurons, fitting [240, 240, 240]) and
schedule (learning rate 1e-3 to 3e-8, energy and force terms), scored on
its 83 sealed-test frames. The hybrid is se_e2_a of the neighbors within
4 Å (`sel` [16, 8, 20, 24, 8]) with se_e2_r of those within 6 Å
([44, 16, 60, 72, 20]): this comparison's own choice, not an example of
DeePMD-kit's. The pipeline marks its models as not production-qualified.

**The same step as DeePMD-kit.** omniMD was given DeePMD-kit's initial
hybrid model and the frames in the order DeePMD-kit took them in a run of
4,000 steps (its sampler's record), with that run's schedule
([`bench/deepmd_replay.py`](../deepmd_replay.py)). It arrives at
DeePMD-kit's model: the embedding nets 1.6e-10 and 4.3e-10 away on average
and the fitting nets 3.2e-10, where DeePMD-kit had moved them by 1.3e-2 to
1.8e-2; the same errors at every step DeePMD-kit displayed, in the three
digits it prints; 0.391268 eV/Å and 45.5012 meV/atom on the sealed test for
both. For what that run has, runs of the two trainers therefore differ by
their random numbers only, the initial networks and the order of the
frames: the energy and force terms with prefactors that follow the learning
rate, a net per neighbor type in both descriptors, Adam and the stepped
schedule, float64, one frame a step. The virial term (off in that request)
and nets per pair of types were not replayed; they rest on finite
differences of omniMD's own loss.

**Speed.** One request, steps per second, the best of three alternated
rounds (20,000 steps in float32, 6,000 in float64); DeePMD-kit 3.1.3
(PyTorch backend, float64, 2,000 steps) on the same request:

| descriptor, embedding nets | parameters | omniMD float32 | omniMD float64 | DeePMD-kit float64 |
|---|---:|---:|---:|---:|
| se_e2_a to 6 Å, a net per pair of types | 2,664,455 | 536 | 249 | 10.4 (2026-11-13) |
| se_e2_a to 6 Å, a net per neighbor type | 2,535,455 | 549 | 250 | 38.8 (2026-11-13) |
| se_e2_r to 6 Å, a net per neighbor type | 735,455 | 1,236 | 434 | 41.3 (2026-11-13) |
| hybrid, the angular descriptor with a net per pair | 2,816,705 | 469 | 216 | |
| hybrid, a net per neighbor type in both | 2,687,705 | 471 | 220 | 27.8 |

- The hybrid trains 17 times faster than DeePMD-kit with float32 networks
  and 7.9 times with float64 ones. Four requests at once run at 149 steps/s
  each.
- It trains 12 to 14 percent slower than se_e2_a to 6 Å, not faster: the
  radial descriptor has that model's slots and nets of the same size, the
  angular one comes on top. A float32 step takes 2.32 ms against 2.04 with
  every stage timed.
- The trainer was laid out by descriptors for this (a model of one
  descriptor is a list of one). Before and after, alternated: se_e2_a 534
  and 537 steps/s in float32 and 246 and 249 in float64, se_e2_r 1,237 and
  1,239 and 436 and 432.

**Against DeePMD-kit, 20,000 steps of the same request.** A net per
neighbor type in both descriptors, four seeds (11 to 14) each, fresh
networks, each trainer's own statistics. Smallest and largest of the four
models over all 83 frames:

| | omniMD float64 | omniMD float32 | DeePMD-kit float64 |
|---|---:|---:|---:|
| force RMSE (eV/Å) | 0.1520 to 0.1714 | 0.1531 to 0.1720 | 0.1401 to 0.1725 |
| energy RMSE (meV/atom) | 8.48 to 16.99 | 10.34 to 15.34 | 7.28 to 11.27 |

- The ranges overlap, in every system as well. DeePMD-kit's four are
  0.140, 0.145, 0.149 and 0.173; eight more omniMD seeds give 0.148 to
  0.166.
- The spread is the schedule's: 20,000 steps stop far from converged, 5,000
  of them at the first learning rate. From one initial model (DeePMD-kit's
  of seed 11), in float32, omniMD's shuffled epochs give 0.155 to 0.166
  (nine training seeds), eight frame sequences drawn by DeePMD-kit's rule
  0.152 to 0.162 and four of independent draws 0.153 to 0.158. DeePMD-kit
  draws a system for every step and is close to independent draws (a frame
  comes again within 20 steps in 9 percent of the cases; 116 different
  frames in 171 steps), where omniMD goes through all frames before one
  comes again; for the result it made no difference here.
- The same frames in the same order do not give the same model twice
  either: DeePMD-kit reached 0.1401 from that initial model with its
  sequence of seed 11, and three replays of that run in omniMD end at
  0.1405, 0.1444 and 0.1478. They follow DeePMD-kit's errors to three
  digits for a thousand steps and then drift: the long stretch at 1e-3
  amplifies the last digits.
- So this comparison cannot tell a difference of a few percent between the
  trainers from none; the replay above can, and finds none. The full
  schedule was not run with DeePMD-kit (three hours a model).
- `omnimd-native-gpu-test` gives `dp test`'s errors of DeePMD-kit's four
  models in all eight systems to the seven digits `dp test` prints.

**The full schedule: is the hybrid worth it on this data?** 300,000 steps,
float32 networks, four seeds at once. Smallest and largest of the four
models over all 83 frames:

| model | force RMSE (eV/Å) | energy RMSE (meV/atom) |
|---|---:|---:|
| se_e2_a to 6 Å (the pipeline's model) | 0.0698 to 0.0723 | 1.64 to 1.78 |
| hybrid, the angular descriptor with a net per pair | 0.0720 to 0.0745 | 1.39 to 1.63 |
| hybrid, a net per neighbor type in both | 0.0737 to 0.0746 | 1.40 to 1.62 |
| se_e2_a to 4 Å alone | 0.0947 to 0.0961 | 1.88 to 2.11 |
| se_e2_r to 6 Å alone | 0.1527 to 0.1574 | 2.72 to 3.75 |

- The radial descriptor to 6 Å takes about a quarter off the force error of
  the angular one to 4 Å and brings it to 3 percent above the angular
  descriptor to 6 Å, which has the lower force error in every system but
  the wet interface. In the energies the hybrid is lower overall, by
  portlandite and lime, and higher in five of the other six systems.
- It buys nothing over se_e2_a to 6 Å here: about its accuracy, for 12 to
  14 percent more time to train and no less to evaluate (see the hybrid
  section below).

Files: `hap-train-speed-hybrid-rtx4090-2026-11-17.json`,
`hap-train-hybrid-against-deepmd-...`, `hap-train-hybrid-replay-deepmd-...`
(the replays, the runs on other frame orders, statistics of the frame
sequences), `hap-train-ensemble-hybrid-float32-omnimd-...`,
`hap-test-deepmd-hybrid-by-omnimd-...`. DeePMD-kit's 20,000-step runs
shared the card with omniMD's, so their own times are not rates; one of
them ended with a device error at step 16,000 when a process beside it was
stopped, and was run again.

# Hybrid descriptors, 2026-11-16

A native model can have DeePMD-kit's `hybrid` descriptor: the components of
several se_e2_a, se_e2_r and se_e3 descriptors, one after the other, are the
input of the fitting nets. Each descriptor of the list has cutoffs, slots,
statistics and nets of its own. As in DeePMD-kit they read one neighbor
list, made with the largest of their cutoffs and for every type the most
slots any of them has: a descriptor has the nearest `sel[t]` of the list's
neighbors of type t in its own slots, and a neighbor at or beyond its own
cutoff is an empty slot to it. It is evaluated on the CPU and on the GPU
runtime (float64 or float32, batches of frames and committees of models,
with embedding tables for its se_e2_a and se_e2_r descriptors if asked) and
converted from DeePMD-kit 3.1.3 `.dp`/YAML by `convert_deepmd.py`. Inference
only on this day: every trainer refused it (the GPU trainer has trained
hybrids since 2026-11-17, those with an se_e3 descriptor included: the
sections above). Format
and formulas are in
[`native/README.md`](../../native/README.md#model-format-v1).

**Agreement with DeePMD-kit 3.1.3** (PyTorch backend, CPU, float64):

- A fixture of 27 atoms of three types under a hybrid of an se_e2_a, an
  se_e2_r and an se_e3 descriptor, no two with the same cutoff, smoothing
  radius, width, slots or statistics (`native/tools/make_hybrid_fixture.py`),
  periodic and without a cell. The CPU evaluation gives DeePMD's energy
  within 1.4e-14 eV, its forces within 4.1e-17 eV/Å (the largest force is
  0.016 eV/Å) and its virial within 2.2e-16 eV. The GPU runtime in float64
  gives the three within 7.1e-15 eV, 2.7e-15 eV/Å and 8.0e-15 eV, a second
  model included (largest force 1.2 eV/Å) whose slots are far below the
  neighbor counts, in the list and in every descriptor. There the result
  depends on which neighbors are kept, and the descriptors of a hybrid do not
  keep the ones each would select alone: the script counts 16, 22 and 18 of
  the 27 atoms for which the three descriptors have other neighbors than
  their own selection would give them.
- The water cases below at 192 and 1,536 atoms (largest force 0.037 and
  0.12 eV/Å: the models are untrained). The se_e2_a + se_e2_r model in
  float64: the same energy to the last digit and forces within 1e-16 eV/Å,
  5e-16 with tables; in float32 within 8.1e-9 eV per atom and 3.6e-8 eV/Å.
  The se_e2_a + se_e3 model: float64 within 9.5e-15 eV per atom and
  2.4e-16 eV/Å, float32 within 5.2e-9 eV per atom and 3.9e-8 eV/Å.

**Water benchmark with two hybrid models.** The sizes are this recipe's own
choice, not an example of DeePMD-kit's (`make_water_case.py --descriptor
hybrid` and `hybrid_e3`): fitting [240, 240, 240] on the components of both
descriptors, untrained, seeded weights, the boxes of the other water cases.
ms per energy, force and virial evaluation. DeePMD: LAMMPS `pair_style
deepmd`, loop time per step, once in NVE dynamics as in the tables of the
other descriptors and once with the atoms held in place (see below). omniMD:
`omnimd-native-gpu-eval` in process, the faster of two runs
(`omnimd-gpu-hybrid-rtx4090-2026-11-16.csv` and `-r2`, `...-hybrid-e3-...`;
the two differ by at most 6%, but for the first row of the first run, which
ran while a container of the run before was being stopped). The speed-up is
against DeePMD with the atoms held.

*Angular near the atom, radial further out*: se_e2_a to 4.0 Å with `sel`
[16, 32] and se_e2_r to 6.0 Å with [46, 92], both with embedding
[25, 50, 100] (1,700 components):

| atoms | precision | DeePMD NVE | DeePMD held | omniMD | speed-up | DeePMD held, tables | omniMD, tables | speed-up |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 192 | fp64 | 14.5 | 14.3 | 1.74 | 8.2× | 11.5 | 0.89 | 13× |
| 1,536 | fp64 | 52.2 | 53.0 | 11.5 | 4.6× | 27.6 | 5.51 | 5.0× |
| 12,288 | fp64 | 370 | 359 | 90.1 | 4.0× | 156 | 41.8 | 3.7× |
| 98,304 | fp64 | error | error | 717 | | error | 333 | |
| 192 | fp32 | 11.4 | 12.3 | 0.38 | 32× | 10.1 | 0.31 | 33× |
| 1,536 | fp32 | 15.8 | 15.9 | 1.67 | 9.5× | 12.2 | 1.11 | 11× |
| 12,288 | fp32 | 89.9 | 83.0 | 10.6 | 7.8× | 36.4 | 6.85 | 5.3× |
| 98,304 | fp32 | error | error | 85.2 | | error | 55.0 | |

*Two-body with three-body*: the se_e2_a descriptor of the plain water model
(6.0 Å, [46, 92], embedding [25, 50, 100]) and se_e3 to 4.0 Å with [16, 32]
and embedding [10, 20, 40] (1,640 components), exact embedding nets on both
sides:

| atoms | precision | DeePMD NVE | DeePMD held | omniMD | speed-up |
|---:|---|---:|---:|---:|---:|
| 192 | fp64 | 25.4 | 25.2 | 2.66 | 9.5× |
| 1,536 | fp64 | 142 | 142 | 18.4 | 7.7× |
| 12,288 | fp64 | no result | no result | 143 | |
| 98,304 | fp64 | error | error | 1,137 | |
| 192 | fp32 | 13.9 | 14.1 | 0.59 | 24× |
| 1,536 | fp32 | 42.1 | 42.5 | 3.36 | 13× |
| 12,288 | fp32 | no result | no result | 23.4 | |
| 98,304 | fp32 | error | error | 186 | |

- **Why the atoms are held.** The benchmark models are untrained and have no
  repulsion. In NVE dynamics the atoms fall into each other after some tens
  to hundreds of femtoseconds, and the steps after that are not steps of the
  benchmark's water. The uncompressed hybrid heats from 300 K to 2,800 K in
  the 0.5 ps of the 192-atom run, with a total energy that is kept for the
  first twenty steps and rises afterwards; its loop time is still that of
  the frames held in place within 9%. omniMD's own NVE run of the same
  model heats the same way (2,682 K after the 1,000 steps, LAMMPS 2,694 K):
  it is the model, and the step is too long for atoms that close. The
  compressed hybrid does not survive it: 81,500 K at 1,536 atoms, and at
  12,288 atoms in float64 2,597 ms a step at 2,850 K, against 195 ms a step
  in the first twenty steps of the same run; the float32 run that followed
  had the card's 24 GB full and was stopped by hand after eight minutes
  (`deepmd-3.1.3-lammps-hybrid-compressed-2026-11-16.csv`, kept as a record
  of that). So DeePMD was timed again with a time step of 1e-6 fs
  (`bench_deepmd.py --held`), in which nothing moves: every step evaluates
  the frame omniMD evaluates. LAMMPS then never rebuilds its neighbor list;
  omniMD searches the neighbors at every evaluation. The logs of the held
  runs were read: 300 K throughout, no neighbor list rebuilt, and an energy
  that stays at the value of the uncompressed model's first step, for the
  compressed model as well (the Python reference, -9975.2494 eV at 192
  atoms).
- The tables of the other descriptors (below) have DeePMD's NVE times. The
  same check for them (`deepmd-3.1.3-lammps-se-e2-a-held-2026-11-16.csv`,
  `-compressed-held`, `-se-e2-r-held`, `-se-e3-held` at 192 atoms): with the
  atoms held DeePMD takes 0.83 to 1.28 times what those tables record, 1.1
  times in the median of twenty rows. Those runs were not slowed by their
  dynamics, and the tables stand.
- At 192 atoms most of DeePMD's step in LAMMPS is a cost that does not grow
  with the atoms (float32: 12.3 ms at 192 atoms, 15.9 ms at 1,536), which
  the in-process evaluator does not have. The multiples of 24 to 33 there
  say that; those at 12,288 atoms (4 to 8) say more about the arithmetic.
- DeePMD-kit's compressed model of the first hybrid is its uncompressed
  model (the same energy, forces within 9e-16 eV/Å on the 192-atom frame, on
  the GPU and on the CPU), so its tables are a baseline. Its compressed
  model of the second hybrid is not: 0.028 eV per atom and 0.23 eV/Å off on
  the GPU in float64, not a number in float32, 0.023 eV per atom off on the
  CPU. That is the se_e3 part again (see the section below), so the second
  table has the exact nets on both sides. omniMD could give the se_e2_a
  descriptor of that model tables and leave the se_e3 nets; that was not
  timed.
- "No result": at 12,288 atoms DeePMD's second hybrid fills the card's
  memory (24.0 of 24.6 GB in use) and gives no result within the 300 s the
  script now allows a run; a first attempt had not finished its first step
  after ten minutes. On Windows such a run is not refused: the driver gives
  it system memory. At 98,304 atoms LAMMPS stops
  with a TorchScript error for both models, as for the other descriptors.
  omniMD takes the atoms in chunks sized from the free memory (two to five
  chunks at 98,304 atoms).
- Where omniMD's time goes, first hybrid in float32: at 192 atoms (0.38 ms)
  the embedding nets 0.12 ms, the fitting nets 0.10, the neighbors 0.08, the
  forces 0.07; at 12,288 atoms (10.6 ms) the embedding nets 5.7 ms, which
  tables bring to 1.6. The table of the list that the se_e2_a descriptor
  reads (its 48 slots of the list's 138) is in the time of the neighbors and
  does not show there: 0.745 ms at 12,288 atoms against 0.743 for the plain
  se_e2_a model.
- This hybrid is no cheaper than the plain se_e2_a model with all 138 slots
  in omniMD (float32, both timed by one script that evening: 0.38 against
  0.31 ms at 192 atoms, 10.6 against 8.7 ms at 12,288; with tables 0.30
  against 0.26 and 6.9 against 6.3): the radial descriptor has embedding
  nets of the same size on the same 138 slots, which is most of the cost of
  either, and the second descriptor adds its launches. In DeePMD-kit it is
  no cheaper either (float32, 12,288 atoms, atoms held: 83.0 against
  69.9 ms).

**How it is built.** The CPU reference computes every descriptor in two
steps, its components and the forces that dE/dD gives, so that a model of
one descriptor and a hybrid go the same way. On the device the potential is
a list of descriptors, each with its nets, statistics, slot tables and
kernels. The neighbors are searched once, with the cutoff and the slots of
the list. A descriptor whose slots or cutoff are not those of the list gets
a table of the list in its own layout from one small kernel (the first
`sel[t]` slots of every type, and no neighbor at or beyond its cutoff),
kept between evaluations; its kernels then run on that table as they run on
the list of a plain model. The descriptor kernels write their components
into one buffer at the descriptor's place among the components of all, the
fitting nets run once on that buffer, and the force kernels read their part
of dE/dD from it; all but the first add to the virial. Chunks of atoms are
common to all descriptors: one ends where its atoms are too many, or its
rows in one of the se_e3 descriptors. A model of one descriptor goes through
the same code as a list of one: for the three plain descriptors the new
build takes what the build before took, within 2.5% in every case of an
alternating comparison (192 to 12,288 atoms, float64, float32 and tables),
and within 0.3% in the two cases furthest apart when they were repeated
eight times.

**Tests.** `native/tests/hybrid.rs` (CPU, runs in CI): DeePMD-kit parity;
forces and all nine strain derivatives against finite differences; with
fitting nets of one linear layer, a hybrid against the sum of models that
have one of its descriptors each (the three kinds in two orders, two of one
kind, one descriptor twice, six descriptors), which sets what a descriptor
reads of the shared list against that descriptor's own search; one
descriptor in a list against that descriptor, bit for bit; the order of the
list; the cutoff of a descriptor against that of the list; a descriptor
with one neighbor more than slots; what the format refuses; the trainers'
refusals. `native/tests/gpu_parity.rs`: hybrids against the CPU reference
(the three kinds; a table of the list for one descriptor and for all; the
slots of the list with a shorter cutoff; a box narrower than the cutoff;
open with three types; embeddings wider than a warp and than a block; a
type without slots; descriptors of one kind and one twice; two se_e3
descriptors in chunks of 1 to 5,000 rows; an atom alone; frames of
different sizes in turn), against DeePMD-kit including the selection when
slots are short, the report of dropped neighbors (one too many, a type
without a slot, a short cutoff, together with atoms in one place), and
float32, batches, committees and tables. `engine/tests/hybrid.rs`: NVE
through the CLI on the CPU (the error of the total energy falls fourfold
when the step is halved) and on the GPU runtime (the CPU trajectory within
1e-10 Å; float32; with tables). A mutation round on the format, the CPU
reference, the kernel of the slot tables, the stride and the virial of
several descriptors, the host side and the committee checks had 70 mutants:
63 were caught by the tests as first written, five showed gaps that now
have checks (a descriptor with exactly one neighbor too many on the CPU;
atoms in one place where a descriptor drops neighbors, which the kernel
must add to the status of the search and not replace; three that only
changed which message a refused committee gets), and two did not compile
and were rewritten. All 70 are caught now. Locally: workspace 219, engine
with `native-gpu` 93, GPU crates 151, each in release and debug; `cargo
fmt --check` clean; the legacy `cuda` features compile. The GPU suites do
not run in CI.

**Not built.** Training of hybrids. Tables for an se_e3 descriptor, alone or
in a hybrid. The embedding nets of all descriptors of a hybrid in one
launch: each descriptor has its own launches.

# The se_e3 descriptor, 2026-11-16

A native model can have DeePMD-kit's three-body descriptor (`se_e3`, also
`se_at` and `se_a_3be`): every pair of an atom's neighbors gives the product
θ of their normalized directions to an embedding net of the pair of neighbor
types, and the descriptor is the sum of θ·G(θ) over the pairs. It is
evaluated on the CPU and on the GPU runtime (float64 or float32, batches of
frames and committees of models, with the exact embedding nets: there are no
tables for it) and converted from DeePMD-kit 3.1.3 `.dp`/YAML by
`convert_deepmd.py`. Inference only at this date: every trainer refused the
descriptor (the GPU trainer has it since 2026-11-17, the first section).
Format and formulas are in
[`native/README.md`](../../native/README.md#model-format-v1).

**Agreement with DeePMD-kit 3.1.3** (PyTorch backend, CPU, float64):

- A fixture of 27 atoms of three types, each type with a width of its own
  for the directions and with a radial mean and width that the descriptor
  must not read (`native/tools/make_se_e3_fixture.py`), periodic and without
  a cell. The CPU evaluation gives DeePMD's energy within 7.1e-15 eV, its
  forces within 9.5e-17 eV/Å (the largest force is 0.05 and 0.08 eV/Å) and
  its virial within 1.2e-15 eV. The GPU runtime in float64 gives the three
  within 7.1e-15 eV, 1.6e-15 eV/Å and 3.6e-15 eV, a second model included
  whose `sel` of [3, 4, 2] is far below the neighbors an atom has there (up
  to 9, 12 and 6 of the three types; largest force 0.8 eV/Å), so that the
  result depends on which neighbors are kept.
- The water case below at 192 atoms (largest force 0.027 eV/Å: the model is
  untrained): float64 within 9.5e-15 eV per atom and 5.6e-17 eV/Å, float32
  within 6.1e-9 eV per atom and 9.3e-9 eV/Å.

**Water benchmark with an se_e3 model**: the cutoff and `sel` of the other
water models (6.0 Å, [46, 92]), the embedding sizes DeePMD-kit's input has by
default for this descriptor ([10, 20, 40]), fitting [240, 240, 240] on the 40
embedding outputs; untrained, seeded weights (`make_water_case.py
--descriptor se_e3`). ms per energy, force and virial evaluation. DeePMD:
LAMMPS `pair_style deepmd`, NVE loop time per step, 2026-11-15
(`deepmd-3.1.3-lammps-se-e3-2026-11-15.csv`). omniMD: `omnimd-native-gpu-eval`
in process, the faster of two runs (`omnimd-gpu-se-e3-rtx4090-2026-11-16.csv`
and `-r2`; the two differ by up to 16% that night, the card was not quiet):

| atoms | precision | DeePMD | omniMD | speed-up |
|---:|---|---:|---:|---:|
| 192 | fp64 | 92.3 | 11.0 | 8.4× |
| 1,536 | fp64 | 6,494 | 86.3 | 75× |
| 12,288 | fp64 | out of memory | 659 | |
| 98,304 | fp64 | error | 5,263 | |
| 192 | fp32 | 28.8 | 2.33 | 12× |
| 1,536 | fp32 | 219 | 15.8 | 14× |
| 12,288 | fp32 | error | 121 | |
| 98,304 | fp32 | error | 994 | |

- The descriptor is of another order of cost than the two-body ones: an
  atom of this water has 84 neighbors on average and 3,559 pairs of them,
  each an evaluation of an embedding net, where se_e2_a has one for each of
  its 138 slots. DeePMD-kit evaluates the nets for all 14,812 pairs of slots
  of an atom, empty slots and both orders of a pair included; omniMD for the
  pairs of neighbors, each once: 4.2 times fewer evaluations in this water.
- DeePMD's float64 step at 1,536 atoms takes seventy times that at 192 for
  eight times the atoms. Probably the run no longer fits the card's own
  memory there (its memory use was not measured); from 12,288 atoms on
  LAMMPS stops with a TorchScript error (reported as out of memory for
  float64 at 12,288 atoms). As before, LAMMPS keeps a neighbor list for
  several steps and omniMD searches at every evaluation.
- A compressed DeePMD-kit model could not be timed: the model that the steps
  of `dp --pt compress` make of this one (`compress_deepmd.py`) is not the
  uncompressed model. On the 192-atom frame (Python API, GPU, float64) its
  energy is 0.012 eV per atom off and its forces up to 0.04 eV/Å, where the
  largest force is 0.027; two compressions with the same settings differ in
  the third decimal of the energy, another minimum distance or stride gives
  another result (0.013 and 0.028 eV per atom), and on the CPU the model
  returns NaN. In LAMMPS the NVE run heats from 305 K to 15,000 K within 200
  steps and loses atoms (192 atoms, float64) or LAMMPS aborts at the first
  evaluation (1,536 atoms); the one run that completed (192 atoms, float32,
  10.9 ms a step with a neighbor list at nearly every step) is not the time
  of a working model. The se_e2_a model compressed by the same script equals
  its uncompressed model within 3e-16 eV/Å on both devices. The rows are in
  `deepmd-3.1.3-lammps-se-e3-compressed-2026-11-16.csv`; the cause inside
  DeePMD-kit 3.1.3 was not looked for.
- omniMD lays the pairs out as rows of the embedding nets, the kernel the
  other descriptors use, and takes the atoms in chunks whose rows fit the
  device memory: 5 chunks at 12,288 atoms in float64 and 3 in float32, 39
  and 20 at 98,304. The rows of a frame are laid out from the number of
  neighbors every atom holds, which costs one small download per evaluation.
- Where the time goes (stage times with a synchronization after each). 192
  atoms, float32, 2.33 ms: embedding nets 1.17 ms, forces 0.41, descriptor
  0.27, the rows of the pairs 0.14, neighbor search 0.12 to 0.16, fitting
  0.14. 98,304 atoms, float32, 994 ms: embedding nets 671 ms, forces 198,
  descriptor 101, rows 35. In float64 the embedding nets are more than four
  fifths (192 atoms: 9.1 of 11.0 ms).
- The descriptor and force kernels read the embedding rows of an atom, and a
  warp waits for every row it reads. In their first version one or two
  warps (descriptor) and four with atomics on shared sums (forces) went
  through an atom's rows: 0.68 and 1.7 ms of then 3.9 ms at 192 atoms in
  float32. Now eight warps share the rows of the descriptor and thirty-two
  those of the forces, which gather their sums neighbor by neighbor without
  atomics: 0.27 and 0.41 ms. A thread per row was tried for the forces: as
  fast at 192 atoms, up to twice slower at 1,536 and more (its loads are
  scattered).
- Not built: tables for these nets (DeePMD-kit has them, with the result
  above), and kernels that contract the rows as they are made instead of
  storing them. At 1,536 atoms in float32 the embedding stage writes 1.7 GB
  of values and tangents, and the descriptor and force kernels, which only
  read them back, take 4.6 of the 15.8 ms.
- Through the engine, `run` takes such a model on the CPU and on the GPU
  runtime like any native model (tests `engine/tests/se_e3.rs`: the GPU run
  follows the CPU trajectory within 1e-10 Å over 60 steps, the error of the
  total energy falls about fourfold when the time step is halved, and a
  request for embedding tables is refused with the reason).

# Training se_e2_r, and one embedding net per neighbor type, 2026-11-13

`omnimd-native-gpu-train` trains the radial descriptor as well, and either
descriptor with one embedding net per neighbor type instead of one per pair
of types (DeePMD's `type_one_side`). Such a net is in the parameters once; a
native model carries the mark (`descriptor.type_one_side`), the converter
sets it for one-sided DeePMD models, and a continued run keeps the nets one.
What the trainer does and how it was checked is in
[`native/GPU_TRAINING.md`](../../native/GPU_TRAINING.md).

**Speed**, steps per second for one request on the 171 frames of the HAP
iteration-0 training (54 to 224 atoms, one frame per step), with the settings
of the comparison of 2026-11-10 below and the descriptor changed (`sel`
[44, 16, 60, 72, 20], embedding [25, 50, 100], fitting [240, 240, 240]).
omniMD: 20,000 steps in float32 and 6,000 in float64, three alternated
rounds that agree within one percent. DeePMD-kit 3.1.3: PyTorch backend,
float64, in its image on the same card with nothing else running, the
trainer's own average time per batch over 2,000 steps of the same request
(`deepmd_train.py`):

| descriptor, embedding nets | parameters | omniMD float32 | omniMD float64 | DeePMD-kit |
|---|---:|---:|---:|---:|
| se_e2_a, a net per pair of types | 2,664,455 | 544 | 252 | 10.4 |
| se_e2_a, a net per neighbor type | 2,535,455 | 555 | 254 | 38.8 |
| se_e2_r, a net per pair of types | 864,455 | 1,236 | 438 | not built |
| se_e2_r, a net per neighbor type | 735,455 | 1,246 | 443 | 41.3 |

- se_e2_r trains 2.3 times faster than se_e2_a in float32 and 1.7 times in
  float64: its fitting nets take 100 inputs instead of 1,600 and nothing
  angular is formed. A float32 step with every stage timed takes 0.98 ms
  against 2.00, and no stage stands out any more (fitting passes 0.14 and
  0.16, embedding nets 0.08 and their reverse pass with the gradients 0.18,
  descriptor 0.05 and its reverse pass 0.08, forces 0.06, tangent 0.05,
  optimizer 0.08, loss 0.05).
- Four requests at once in float32 run at 428 steps/s each for se_e2_r with
  a net per pair (1,713 in total, 39 percent more than one request alone),
  441 each with a net per neighbor type, and 152 each for se_e2_a (607 in
  total, 12 percent more). Four se_e2_r models of 300,000 steps took 11.4
  minutes that way (below).
- The trainer of se_e2_a is as fast as before this change: one request,
  alternated with the build of 2026-11-11, 546 against 548 steps/s in
  float32 and 253 against 254 in float64; four at once 152 against 133 (the
  older build's run overlapped the end of another for its first half
  minute). The se_e2_a ensemble of 2026-11-10 ran at 169 each on these
  frames, and the one-request table of that day, 604 and 289, is on 196
  frames with more of the small lime cell.
- omniMD's speed hardly depends on the number of embedding nets (2 percent),
  DeePMD-kit's does: with one net per neighbor type (5 nets instead of 25)
  its se_e2_a training is 3.7 times faster. The pipeline's models have a net
  per pair.
- Against DeePMD-kit on the same request: se_e2_r 11 times faster in float64
  and 30 times with float32 networks; se_e2_a with a net per neighbor type
  6.5 and 14 times; se_e2_a with a net per pair 24 and 52 times.
- The first row is the model of the 2026-11-10 comparison. DeePMD-kit's 10.4
  steps/s stand against 12.3 in the pipeline's record of its own training;
  its thread and allocator settings (the pipeline's, or none) made no
  difference to the se_e2_r run, 0.0242 s per batch both times.
- "Not built": DeePMD-kit 3.1.3's PyTorch backend gives se_e2_r one net per
  neighbor type whatever `type_one_side` says (its input default is false).
  omniMD builds what the key says and prints a note when that differs.

**The same request in both trainers.** se_e2_r with a net per neighbor type
(what DeePMD-kit's PyTorch backend builds), 20,000 steps of one frame, four
seeds (11 to 14) for each trainer; DeePMD-kit 3.1.3 in float64 on the request
files omniMD ran. All models scored on the pipeline's 83-frame sealed test
by `omnimd-native-gpu-test` (`score_groups.py`); smallest and largest of the
four:

| | omniMD float64 | omniMD float32 networks | DeePMD-kit |
|---|---:|---:|---:|
| sealed test, force RMSE (eV/Å) | 0.3040 to 0.3517 | 0.3259 to 0.3445 | 0.3384 to 0.3471 |
| sealed test, energy RMSE (meV/atom) | 9.36 to 22.96 | 9.56 to 19.44 | 10.33 to 19.13 |
| time for the four models | 2.7 min at once | 2.1 min at once | 34.3 min one after the other |

- Same quality: the ranges overlap overall and in each of the eight systems,
  for forces and for energies (per system in
  [`native/GPU_TRAINING.md`](../../native/GPU_TRAINING.md)). 20,000 steps
  leave this model far from converged for both.
- The times are those of these runs: omniMD's two sets of four shared the
  card with test runs, DeePMD-kit's models took 8.0 to 9.6 minutes each (the
  last two alone on the machine, at 41.7 and 42.2 steps/s). The clean rates
  are in the table above.
- The instrument is the same for both: `omnimd-native-gpu-test` gives the
  energy and force RMSE of `dp test` for DeePMD-kit's four models, converted,
  in all eight systems, 64 numbers, to the seven digits `dp test` prints
  (`hap-test-deepmd-se-r-by-omnimd-rtx4090-2026-11-13.json`).
- A model trained by DeePMD-kit continues in omniMD with its nets shared
  (735,455 parameters): after 5,000 more steps at the last learning rate its
  sealed-test force RMSE is 0.3385, from 0.3384.
- The training frames went to DeePMD-kit as npy systems written back from
  omniMD's dataset (`dataset_to_deepmd.py`): they equal the pipeline's
  source arrays number for number.

**What the radial descriptor gives on this data.** The full schedule of the
2026-11-10 comparison (300,000 steps of one frame, learning rate 1e-3 to
3e-8, energy and force terms, float32 networks, seeds 11 to 14 trained at
once) with se_e2_r in place of se_e2_a, scored on the pipeline's 83-frame
sealed test with `omnimd-native-gpu-test`; smallest and largest of the four
models:

| | se_e2_r, a net per pair | se_e2_r, a net per neighbor type | se_e2_a, a net per pair (2026-11-10) |
|---|---:|---:|---:|
| parameters | 864,455 | 735,455 | 2,664,455 |
| sealed test, force RMSE (eV/Å) | 0.1553 to 0.1829 | 0.1527 to 0.1574 | 0.0698 to 0.0723 |
| sealed test, energy RMSE (meV/atom) | 3.13 to 4.51 | 2.72 to 3.75 | 1.64 to 1.78 |
| time for the four models, at once | 14.9 min beside test runs | 11.4 min | 30 min |

- se_e2_r trains 2.3 times faster and is less accurate in the forces of
  every system, 2.1 to 2.6 times overall (per system in
  [`native/GPU_TRAINING.md`](../../native/GPU_TRAINING.md)): without angular
  information it is not a substitute for se_e2_a here.
- Sharing the nets between the central types costs nothing here: the force
  RMSE is as low or lower, and the seeds spread by 3 percent in it where the
  nets of the pairs spread by 18 (se_e2_a: 3.6).
- The se_e2_a models are those of 2026-11-10, scored again: the same numbers.

Raw numbers: `hap-train-speed-descriptors-rtx4090-2026-11-13.json` (rates,
stage times, DeePMD-kit's own timings),
`hap-train-se-r-against-deepmd-rtx4090-2026-11-13.json` (the 20,000-step
models of both trainers),
`hap-test-deepmd-se-r-by-omnimd-rtx4090-2026-11-13.json` (`dp test` against
`omnimd-native-gpu-test`) and
`hap-train-ensemble-se-r-float32-omnimd-rtx4090-2026-11-13.json` (the full
schedule).

# The se_e2_r descriptor, 2026-11-12

A native model can have DeePMD-kit's radial descriptor (`se_e2_r`, also
`se_r`): a fifth of the mean of the embedding rows of an atom's neighbor
slots, without the angular part of se_e2_a. It is evaluated on the CPU and on
the GPU runtime (float64 or float32, exact embedding nets or tables, batches
of frames and committees of models) and converted from DeePMD-kit 3.1.3
`.dp`/YAML by `convert_deepmd.py`. This section is inference; training came
a day later and is the section above. Format and formulas are in
[`native/README.md`](../../native/README.md#model-format-v1).

**Agreement with DeePMD-kit 3.1.3** (PyTorch backend, CPU, float64):

- A fixture of 27 atoms of two types with a normalization of its own for
  each (`native/tools/make_se_r_fixture.py`), periodic and without a cell.
  The CPU evaluation gives DeePMD's energy within 7e-15 eV, its forces within
  1e-17 eV/Å (the largest force is 7e-3 eV/Å) and its virial within 1.2e-16
  eV. The GPU runtime in float64 passes the same comparison, and that with a
  second model whose `sel` of [4, 6] is far below the some 23 neighbors of
  an atom, where the result depends on which neighbors are kept.
- The water cases below at 192 and 1,536 atoms: float64 within 9.5e-15 eV per
  atom and 1.2e-18 eV/Å (largest force 2.7e-4 eV/Å: the model is untrained),
  with tables 2e-16 eV/Å, float32 within 6.7e-9 eV per atom and 5.9e-10 eV/Å.

**Water benchmark with an se_e2_r model** of the sizes of the se_e2_a one
(rcut 6.0 Å, sel [46, 92], embedding [25, 50, 100], fitting [240, 240, 240]
on the 100 embedding outputs; untrained, seeded weights; `make_water_case.py
--descriptor se_e2_r`). ms per energy, force and virial evaluation. DeePMD:
LAMMPS `pair_style deepmd`, NVE loop time per step
(`deepmd-3.1.3-lammps-se-r-2026-11-12.csv`). omniMD: `omnimd-native-gpu-eval`
in process (`omnimd-gpu-se-r-rtx4090-2026-11-12.csv`, one run each; a second
run agreed within 3%):

| atoms | precision | DeePMD | omniMD | speed-up | omniMD, tables |
|---:|---|---:|---:|---:|---:|
| 192 | fp64 | 5.90 | 0.98 | 6.0× | 0.36 |
| 1,536 | fp64 | 26.3 | 6.03 | 4.4× | 2.12 |
| 12,288 | fp64 | 193 | 47.3 | 4.1× | 15.7 |
| 98,304 | fp64 | error | 377 | | 125 |
| 192 | fp32 | 4.96 | 0.26 | 19× | 0.20 |
| 1,536 | fp32 | 7.38 | 1.16 | 6.4× | 0.80 |
| 12,288 | fp32 | 52.9 | 6.93 | 7.6× | 4.24 |
| 98,304 | fp32 | error | 55.1 | | 34.8 |

- As in the se_e2_a comparison, LAMMPS keeps a neighbor list for several
  steps and omniMD searches at every evaluation, and LAMMPS stops at 98,304
  atoms with a TorchScript error. A compressed DeePMD model of this
  descriptor was not timed.
- Against se_e2_a in omniMD, same script, build and day
  (`omnimd-gpu-rtx4090-2026-11-12.csv`): float64 0.98 against 1.46 ms at 192
  atoms and 47.3 against 68.9 at 12,288; float32 with tables 0.20 against
  0.25, 0.80 against 0.93, 4.24 against 5.80 and 34.8 against 45.9 ms.
- Where the time goes at 98,304 atoms in float32 with tables (34.8 ms; stage
  times with a synchronization after each): embedding tables 9.3 ms, neighbor
  search 6.3, forces 5.6, fitting 4.7, descriptor 4.5. The descriptor only
  adds up embedding rows, and the force kernel only contracts their
  derivatives with one vector per atom: both read from memory what the
  embedding stage has just written there. A kernel that summed the rows as
  it made them would not have to store them; that is not built.
- Through the engine, `run` takes such a model on the CPU and on the GPU
  runtime like any native model (tests `engine/tests/se_e2_r.rs`: the GPU
  run follows the CPU trajectory within 1e-10 Å over 60 steps, and the error
  of the total energy falls about fourfold when the time step is halved).

# Float32 models without double precision between the environment and its adjoint, 2026-11-11

A float32 model ran its networks in single precision and everything around
them in double: the environment rows, the sums that make B, the descriptor
and the first part of the force kernel. On this card double precision has a
sixty-fourth of the rate of single, and the descriptor kernel does little
else: four multiply-adds for every neighbor and embedding output.

Now the potential keeps what lies between the environment matrix and its
adjoint in the precision of the networks: the environment rows as they are
stored, B, the descriptor and the adjoint of B. Geometry stays double: the
rows are worked out in double before they are stored, and their Jacobian
with the sums that give forces and virial is double. A float64 model
computes what it did (the sums of B run type by type now, so its last digits
moved); the trainer keeps double precision there (the kernels take the
precision as a parameter). The descriptor kernel also copies an atom's
environment rows and its number of neighbors of each type to shared memory
before it sums them.

**Passes over copies of one HAP frame**, microseconds per pass, float32 with
compressed embedding, the two builds alternated four times and the best run
of each kept (`omnimd-native-gpu-eval --bench 1500 --bench-batches
1,2,4,8,16`):

| frames | 88 atoms, before | now | 169 atoms, before | now | 224 atoms, before | now |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 197 | 178 | 253 | 215 | 272 | 238 |
| 2 | 249 | 215 | 362 | 305 | 459 | 380 |
| 4 | 361 | 307 | 655 | 536 | 824 | 677 |
| 8 | 666 | 554 | 1262 | 1031 | 1545 | 1347 |
| 16 | 1270 | 1043 | 2108 | 1843 | 2498 | 2105 |

**Water**, ms per evaluation (`--bench K`, alternated three times, best):

| atoms | fp32 compressed, before | now | fp32, before | now | fp64 compressed, before | now | fp64, before | now |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 192 | 0.279 | 0.242 | 0.328 | 0.291 | 0.797 | 0.794 | 1.319 | 1.315 |
| 1,536 | 1.144 | 0.936 | 1.531 | 1.326 | 4.874 | 4.866 | 8.804 | 8.807 |
| 12,288 | 7.01 | 5.66 | 9.59 | 8.33 | 37.00 | 37.01 | 68.37 | 68.40 |
| 98,304 | 56.1 | 45.5 | 76.5 | 66.4 | | | | |

- A float32 pass is 10 to 19% faster at every size; float64 is unchanged.
  With compressed embedding the 12,288 atoms take 5.7 ms where LAMMPS
  `pair_style deepmd` with the compressed float32 model took 32.5 (the
  record of 2026-11-10 below).
- **The exploration of the section below** (16 runs of 40,000 steps, four
  models compared every 50 steps): 80.0 s of stepping with the previous
  build run again, 69.6 s now, 9,198 run-steps per second
  (`hap-explore-all-rtx4090-2026-11-11-r2.json`).
- **Agreement with DeePMD-kit** on the 83 HAP test frames (`hap_check.py`,
  largest difference over the four models, before and now):

  | mode | energy per atom (eV) | force (eV/Å) | virial (eV) |
  |---|---:|---:|---:|
  | float64 | 1.8e-13, 1.8e-13 | 5.6e-13, 5.6e-13 | 1.1e-12, 1.1e-12 |
  | float64, compressed | 8.4e-10, 8.4e-10 | 3.9e-6, 3.9e-6 | 2.0e-5, 2.0e-5 |
  | float32 | 2.4e-7, 3.4e-7 | 7.7e-6, 8.9e-6 | 7.9e-5, 1.1e-4 |
  | float32, compressed | 3.7e-7, 4.4e-7 | 7.3e-6, 8.7e-6 | 7.3e-5, 8.1e-5 |

  The float32 modes move by a factor of 1.1 to 1.4, within the order they
  had: a sum of some hundred products in single precision adds about what
  the single-precision networks already did. They remain three to four
  orders of magnitude below the models' own test errors (0.014 to 0.098
  eV/Å).
- Where the time goes at 98,304 atoms now (45.5 ms; stage times with a
  synchronization after each): forces 11.4 ms (15.5 before), embedding tables
  9.4, fitting 8.5, neighbor search 6.1, descriptor 5.6 (11.9 before). The
  descriptor reads 3.5 GB of embedding rows in those 5.6 ms (98,304 atoms
  with 89.5 neighbors each and 100 values a neighbor): it moves at the rate
  of the memory now, as do the embedding stage that writes those rows and
  the force kernel that reads them again. Copying the environment rows to shared
  memory changed nothing there; for small systems, where the time of a
  launch is the latency of its loads, it took another 5% off the pass (88
  atoms: 190 to 180 microseconds, the two variants alternated).
- Tests: an embedding wider than a warp and a type without neighbor slots
  were added to the comparison with the CPU reference (the first because two
  mutants of the new shared-memory layout survived until a kernel had more
  than one warp of threads at work).

# An exploration in one run, and its candidates, 2026-11-11

`collect-candidates` reads the trajectories of runs that carried the members
of their ensemble along and writes the candidate frames of all of them as one
dataset, at most so many of a run and those evenly spread in time, with a
summary of the frames the members agree on
([`engine/MODEL_DEVIATION.md`](../../engine/MODEL_DEVIATION.md#8-the-candidates-of-an-exploration)).
With `run-replicas` that makes an exploration two commands.

**An exploration with the pipeline's protocol** (`hap_explore_all.py`): the
eight systems of the HAP case (54 to 224 atoms) from their first frame at 300
and at 500 K, so 16 runs and 1,912 atoms a step; 40,000 steps of 0.5 fs each;
model 0 driving and all four models compared every 50 steps; window 0.10 to
0.30 eV/Å, at most 30 candidates a run. Compressed float32, CSVR thermostat.
In the second column the six bulk systems run at 300 K at 1.01325 bar with a
fully flexible cell, as the pipeline explores them from its iteration 4 on.
Raw numbers in `hap-explore-all-rtx4090-2026-11-11.json` and
`hap-explore-all-npt-rtx4090-2026-11-11.json`:

| | fixed cells | bulk at 300 K at constant pressure |
|---|---:|---:|
| stepping, 16 runs of 40,000 steps | 79.2 s | 82.7 s |
| one step of all 16 runs | 1.98 ms | 2.07 ms |
| run-steps per second | 8,077 | 7,743 |
| the `run-replicas` process in all | 81.7 s | 84.9 s |
| `collect-candidates` | 1.9 s | 1.9 s |
| sampled frames | 12,816 | 12,816 |
| accurate, candidate, failed | 11,360 / 1,448 / 8 | 11,430 / 1,377 / 9 |
| picked | 144 | 141 |

- For scale, two records of LAMMPS `pair_style deepmd` with the same four
  models and the same sampling (the section of 2026-11-10 below): one run
  alone on this card took 12.8 to 16.5 ms a step, 61 to 78 run-steps per
  second; the pipeline's own logs show 52 ms a step with three runs sharing
  the card, 58 run-steps per second in all. The 640,000 run-steps of this
  exploration are about three hours at those rates and 79 seconds here, a
  factor of 100 to 140. The thermostats differ (Nose-Hoover there), which
  does not affect the cost of a step.
- The pipeline explores 36 runs an iteration since it added further phases;
  the case here holds the eight systems of its sealed test.
- Constant pressure for six of the sixteen runs costs 4%: every ten steps
  their cell moves take one more evaluation, of those six systems together.
- Outside the stepping loop the process spends 2.5 s, mostly on loading four
  95 MB models and building their tables. The sixteen trajectories take 300
  MB; reading them twice for the candidates takes 1.9 s.
- The 500 K runs are the same requests in both columns and count
  differently: reductions on the device are not ordered, so two runs of one
  request agree to rounding at first and part ways like any two trajectories
  of a chaotic system.
- The counts say how far four models of one round agree along these
  particular runs. They are not a statement about the accuracy of the
  models, which the pipeline marks as not production-qualified.

# Fitting nets: a weight read serves two rows, 2026-11-11

The batch timings below end in a plateau: between some 300 and 1,400 atoms a
pass, the fused fitting kernels were the largest stage and cost the same per
row whatever the rows. Taken apart by leaving out one launch at a time, the
stage of 2,816 rows was 0.07 microseconds a row in the forward tiles of the
first layer, 0.20 in the layers after it and 0.19 in the backward tiles.

Two changes, both in `mlmd/gpu/src/fused.rs`: the layers after the first take
two rows a block, so that a weight read from global memory serves both; and
the tile of the first layer's transpose does two rows a warp, as the forward
tile already did. Why two and not more is in
[`gpu/README.md`](../../gpu/README.md).

**The fitting stage alone**, microseconds per call for 1600 -> 240 -> 240 ->
240 -> 1 nets in float32, rows in the proportions of the HAP bulk frame (best
of four, the variants alternated in one process; the probe `fitting_costs`
of `gpu/tests/costs.rs` prints the column "now" as "tiles"):

| rows | before | now |
|---:|---:|---:|
| 88 | 75 | 67 |
| 176 | 108 | 86 |
| 352 | 166 | 148 |
| 704 | 341 | 255 |
| 1,408 | 655 | 496 |
| 2,816 | 1331 | 1052 |

**Whole passes**, microseconds per pass over copies of one frame, float32
with compressed embedding, the two builds alternated three times and the
best run of each kept (`omnimd-native-gpu-eval --bench 1500 --bench-batches
1,2,4,8,16`):

| frames | 88 atoms, before | now | 169 atoms, before | now | 224 atoms, before | now |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 208 | 201 | 272 | 254 | 307 | 275 |
| 2 | 272 | 249 | 391 | 363 | 512 | 461 |
| 4 | 390 | 362 | 755 | 664 | 925 | 822 |
| 8 | 781 | 677 | 1501 | 1276 | 1568 | 1578 |
| 16 | 1387 | 1286 | 2154 | 2153 | 2515 | 2522 |

- A single frame is 3 to 10% faster, a pass of two to eight frames 7 to 15%
  as long as it stays on the fused kernels.
- With the fused kernels faster, they now meet the layer-by-layer cuBLAS path
  near 340 atoms a type and model instead of 256 (measured with five types
  at 1,800 to 2,000 atoms a pass and with two types at 680), and the switch
  moved there. That is the gain at 8 frames of 169 atoms and at 16 of 88;
  beyond the switch a pass is cuBLAS as before (8 and 16 frames of 224 atoms,
  16 of 169: equal within the noise of the measurement).
- What is left of the fitting stage with many rows: 0.07 microseconds a row
  in the forward tiles, 0.14 in the layers after the first, 0.15 in the
  backward tiles.
- The tables below were measured before this change. Their entries for
  passes of up to some 1,700 atoms in float32 are that much lower now; the
  float64 ones are as they were (the tiles are a float32 path).

# Model deviation along a run, 2026-11-11

A `run` or `run-replicas` request can name the other members of its model's
ensemble (`deviation`). The model of the run drives the dynamics; at every
sampled frame all members are evaluated, as one committee on the GPU runtime,
and the frame is written with their deviation and whether that makes it a
candidate. `import-trajectory` can then keep the candidates only. Exploration
and the comparison of the members were two passes over every frame before
(run, import, `model-deviation`); they are one now.

**Microseconds per step and replica** (`hap_replicas.py --deviation`: NVT at
300 K, 0.5 fs, sampled every 50 steps; the four HAP models, model 0 driving;
least of two runs; raw numbers in `hap-run-deviation-rtx4090-2026-11-11.csv`):

| system (atoms) | mode | run | with the members | 16 replicas | with the members |
|---|---|---:|---:|---:|---:|
| HAP bulk (88) | float32, compressed | 212 | 225 | 88 | 95 |
| | float64 | 738 | 786 | 525 | 563 |
| HAP(001) + water + CO2 (169) | float32, compressed | 285 | 302 | 143 | 152 |
| | float64 | 1271 | 1349 | 984 | 1062 |
| monetite (224) | float32, compressed | 317 | 339 | 169 | 181 |
| | float64 | 1539 | 1654 | 1365 | 1475 |

- The members cost 6 to 8% of a run when one frame in fifty is sampled: three
  more model evaluations per fifty steps, all in the pass that the sampled
  frame costs anyway.
- For scale: LAMMPS `pair_style deepmd` with the same four models and the
  same sampling (model 0 drives, all four are evaluated every 50 steps) took
  12.8, 16.5 and 15.5 ms per step for these systems on this card (the table
  of 2026-11-10 below; Nose-Hoover there, CSVR here). That is 9 to 16 times
  the float64 column with the members and 46 to 57 times the compressed
  float32 one.
- On the CPU the deviation of every sampled frame equals, to the last digit,
  what `model-deviation` computes from the imported trajectory, and the run
  itself is the run without the members (tests `run_deviation.rs`). On the
  GPU runtime the frames' deviations agree with the CPU's within 1e-9 eV/Å in
  float64 (`gpu_runtime.rs`).

# Several runs advanced together, and a lighter step, 2026-11-11

Two changes to molecular dynamics through `omnimd-mlmd`, both on the host.

**Replicas.** `run-replicas` advances several independent systems of one
model in step: one model evaluation per step for all of them, which the GPU
runtime answers in one pass over the device (the batches of the section
below), and one more for the systems whose cell moves at that step. Each
replica has its own ensemble and writes the trajectory and checkpoints it
writes alone; on the CPU they are identical to those of the single runs.

**The step itself.** A step used to copy the whole system, names and
molecules included, and to build the frame handed to the model five times
over; it now works on the system in place and takes the step back if any
part of it fails. The host's share of a step, measured with a model that
costs nothing (`tests/step_cost.rs`):

| atoms | NVE step, before | now | NVT step, before | now |
|---:|---:|---:|---:|---:|
| 88 | 13.4 µs | 1.0 µs | 20.7 µs | 1.75 µs |
| 224 | 35.7 | 2.5 | 50.7 | 3.9 |
| 1,024 | 165 | 11.6 | 222 | 16.3 |

Trajectories and checkpoints of the CPU path are the same bytes as before
(NVE, NVT and NPT runs of 400 steps compared between the two builds).

**Microseconds per step and replica** (`hap_replicas.py`: NVT at 300 K, CSVR
thermostat, 0.5 fs, sampled every 50 steps, model 0 of the HAP ensemble;
replicas of one system with different seeds; the engine's own count of its
stepping time, least of two runs; raw numbers in
`hap-replicas-rtx4090-2026-11-11.csv`). Float32 networks with compressed
embedding:

| system (atoms) | plain run, before | plain run | 2 replicas | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HAP bulk (88) | 235 | 215 | 144 | 102 | 104 | 90 | 71 |
| HAP(001) + water + CO2 (169) | 320 | 285 | 208 | 197 | 200 | 146 | 121 |
| monetite (224) | 361 | 320 | 271 | 242 | 208 | 171 | 138 |

Float64 networks:

| system (atoms) | plain run, before | plain run | 2 replicas | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HAP bulk (88) | 764 | 744 | 653 | 551 | 534 | 525 | 522 |
| HAP(001) + water + CO2 (169) | 1310 | 1272 | 1063 | 1039 | 1014 | 988 | 973 |
| monetite (224) | 1587 | 1544 | 1578 | 1447 | 1392 | 1372 | 1355 |

- A plain run is 8 to 11% faster in compressed float32 (20 to 41 µs less a
  step) and about 3% faster in float64, from the lighter step alone. ("Before"
  is the previous build timed by the difference of two runs of different
  length; that build did not report its stepping time.)
- Replicas in compressed float32: sixteen 88-atom systems advance at 90 µs
  per step each and thirty-two at 71 µs, against 215 µs alone, that is 2.4
  and 3.0 times the simulated time per day from one card (19,400 ns per day
  in all for thirty-two replicas at 0.5 fs, against 200 for one run). For
  169 and 224 atoms the gain is 2.0 and 1.9 times at sixteen replicas and 2.4
  and 2.3 times at thirty-two.
- The engine adds some 5 to 15 µs per replica and step to the evaluation of
  the batch (integrator, building frames, sampled output), so the table follows
  the batch timings below; the plateau between 300 and 1,400 atoms per pass
  is the fused fitting kernel, as there.
- Float64 gains 1.1 to 1.4 times, as in the batch timings.
- GPU replicas against CPU replicas of the same request (three systems, NVE,
  NPT with cell moves, NVT without a cell; test `gpu_runtime.rs`): sampled
  frames within 1e-9 over 40 steps in float64.

# Batches of frames and committees of models, 2026-11-11

A system of a hundred atoms leaves most of the card idle: its evaluation is a
dozen launches of a few microseconds that wait for each other. The GPU
se_e2_a path now takes more work per launch in two ways (layout in
[`gpu/README.md`](../../gpu/README.md)). A batch of frames is one list of
atoms on the device, with a neighbor search that keeps the frames apart; and
models of one architecture, such as an ensemble kept for model deviation, are
a second dimension of every launch, with one neighbor search for all of them.

**Microseconds per frame and model** in process (`hap_batch.py`, which runs
`omnimd-native-gpu-eval --bench-batches` over copies of one frame; least of
three runs; raw numbers in `hap-batch-rtx4090-2026-11-11.csv`). HAP models,
float32 networks with compressed embedding:

| system (atoms) | models | 1 frame | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HAP bulk (88) | 1 | 206 | 134 | 96 | 98 | 85 | 65 | 56 |
| | 4 | 125 | 97 | 87 | 84 | 76 | 60 | 49 |
| HAP(001) + water + CO2 (169) | 1 | 272 | 196 | 190 | 183 | 134 | 110 | 99 |
| | 4 | 199 | 174 | 161 | 164 | 122 | 98 | 88 |
| monetite (224) | 1 | 306 | 259 | 231 | 193 | 158 | 130 | 130 |
| | 4 | 233 | 221 | 208 | 176 | 139 | 123 | 119 |

Float64 networks:

| system (atoms) | models | 1 frame | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HAP bulk (88) | 1 | 733 | 643 | 542 | 521 | 521 | 515 | 504 |
| | 4 | 539 | 522 | 517 | 511 | 499 | 498 | 496 |
| HAP(001) + water + CO2 (169) | 1 | 1249 | 1045 | 1023 | 986 | 958 | 948 | 941 |
| | 4 | 1006 | 970 | 952 | 939 | 931 | 927 | 926 |
| monetite (224) | 1 | 1522 | 1563 | 1428 | 1372 | 1351 | 1337 | 1329 |
| | 4 | 1409 | 1356 | 1347 | 1325 | 1315 | 1314 | 1313 |

- **Batches** (one model, float32 compressed): a frame in a batch of 16
  costs 0.41 of a frame alone for 88 atoms, 0.49 for 169 and 0.52 for 224;
  in a batch of 64, 0.27, 0.36 and 0.42. Sixteen copies of the 88-atom
  system are evaluated in 1.36 ms, against 3.3 ms one after the other.
- **Committees**: the four models on one frame, which is what model deviation
  along a trajectory asks for, cost 0.50 ms for 88 atoms against 0.82 ms one
  by one (1.65 times faster), 0.80 against 1.09 ms for 169 atoms and 0.93
  against 1.22 ms for 224.
- **Float64** gains 1.15 to 1.45 times: its arithmetic already keeps the
  double precision units of the card busy for a single frame, so only the
  launches and the shared neighbor search are saved.
- Where the gain ends. With more atoms in a pass the launches no longer
  count and the stages are bound by what they compute: for 8 frames of 88
  atoms the fused fitting kernel is 44% of the pass (its layers after the
  first read their weights a row at a time), the force kernel 20%, the
  descriptor 14%. Above 256 atoms per type and model the fitting nets go
  layer by layer through cuBLAS, which costs 0.68 ms in launches for five
  types whatever the rows and little more for more rows; the two paths
  cross near 1,400 atoms as before, and between some 300 and 1,400 atoms a
  pass gains little per frame (the plateau at 2 to 8 frames of 169 atoms).
  That range is the next thing to work on.
- **Agreement.** The four models as one committee over the 83 HAP test
  frames in batches of 16 (`hap_batch.py --check`), each member against
  DeePMD's own evaluation with that model: float64 within 1.8e-13 eV per
  atom and 5.6e-13 eV/Å, float64 compressed 8.4e-10 and 3.9e-6, float32
  3.1e-7 and 8.5e-6, float32 compressed 4.5e-7 and 1.0e-5. The float64
  numbers are those of the single model; the float32 ones differ in the
  last digits because batches of that size take the cuBLAS fitting path. A
  batch gives each frame the neighbors it gets alone, bit for bit (tested
  against each system searched alone).

**Model deviation through the engine** (`hap_deviation.py`:
`omnimd-mlmd model-deviation`, the four models on `cuda:0`, the 83 frames of
54 to 224 atoms; time per frame from two runs over the frames once and twelve
times, which takes out loading; this build against the one before, which
evaluated member by member and frame by frame):

| mode | before, ms per frame | now | frames per second |
|---|---:|---:|---:|
| float64 | 3.79 | 2.42 | 1.57 times |
| float32, compressed | 1.17 | 0.47 | 2.48 times |

The deviations of the two builds agree to 3e-15 eV/Å in float64 (1e-6 in
compressed float32) and with those computed from DeePMD's reference forces to
1e-14 eV/Å (4e-6). A whole run over the 83 frames took 1.7 s in float64 and
4.9 s compressed: loading four models, and for the compressed mode building
their embedding tables, about 0.8 s per model on one thread. The tables are
now built on all cores: a process that loads one model, builds its tables and
evaluates a frame takes 0.68 s instead of 1.40 s (0.48 s without tables), and
with the four models as a committee 1.69 s instead of 4.98 s (0.94 s without).

`batch-eval` reads a dataset the same way, a batch of frames per pass.

# Fitting nets with the first layer in tiles, 2026-11-11

After the rework below the fitting nets were 40 to 60% of a small system's
evaluation, bound by reading their weights from global memory a row at a
time. For rows grouped by network (`GradientNetwork::forward_gradient_grouped`)
the first layer, 1600 to 240 here and three quarters of the arithmetic, now
goes through shared memory in tiles that serve all rows of a chunk; the other
layers stay row by row. Layout and measurements are in
[`gpu/README.md`](../../gpu/README.md).

**Microseconds per evaluation** in process (`omnimd-native-gpu-eval --bench
2500`), same HAP model and frames, the previous build and this one alternated
four times and the best run of each kept:

| system (atoms) | fp32 | fp32, tiles | fp32 compressed | fp32 compressed, tiles |
|---|---:|---:|---:|---:|
| HAP bulk (88) | 269 | 230 | 244 | 205 |
| HAP(001) + water + CO2 (169) | 473 | 368 | 402 | 301 |
| monetite (224) | 552 | 421 | 477 | 356 |

- Float32 evaluations are 1.17 to 1.34 times faster. Float64 is unchanged:
  there the arithmetic is the bound and the tiles measured slower (88 atoms,
  compressed: 457 to 503), so they stay off.
- The fitting kernels alone, 1600 -> 240 -> 240 -> 240 -> 1 nets, per call
  (`fitting_costs` probe): 88 rows in four type runs 125 to 81 µs, 224 rows
  270 to 149, 1024 rows of one type 757 to 498.
- Agreement with DeePMD on the 83 HAP test frames (`hap_check.py`): float32
  within 2.4e-7 eV per atom and 7.5e-6 eV/Å as before; float64 unchanged.
- Why best runs: the card is shared with the desktop (an animated wallpaper
  takes up to a fifth of it), single runs of one build differ by 10 to 25%
  and the best of four by about 2% (the float64 modes, whose code did not
  change, came out within 2% of their previous value). Whole-process
  timings of molecular dynamics through the engine scatter too much under
  that load to resolve this gain, so the table of steps below stands as the
  last clean record of those.
- The trainer, measured again on the idle card for the rework below (it
  shares those kernels; it does not use the tiled ones): float32 597 to 622,
  626 to 649 and 688 to 700 steps/s for 1, 2 and 4 requests at once; float64
  unchanged within the scatter (best of two alternated rounds 240 and 282,
  304 and 308, 333 and 336).

# Evaluation after the call and double-precision rework, 2026-11-10 (r3)

The GPU se_e2_a path was measured piece by piece (`mlmd/gpu/tests/costs.rs`,
results and consequences in [`gpu/README.md`](../../gpu/README.md)) and
rebuilt around what that showed: a host call costs 5 to 10 microseconds, a
division in double precision as much as sixty multiplications, and most
neighbor slots are empty. Same models, frames and protocols as the tables
further down; the previous build was run again in the same session.

**ms per MD step**, NVT at 300 K through `omnimd-mlmd`
(`hap-omnimd-rtx4090-2026-11-10-r2.csv`), LAMMPS from the record below:

| system (atoms) | LAMMPS `pair_style deepmd` | fp64 | fp64 compressed | fp32 | fp32 compressed |
|---|---:|---:|---:|---:|---:|
| HAP bulk (88), previous build | | 1.60 | 0.84 | 0.58 | 0.47 |
| HAP bulk (88) | 13.6 | 0.80 | 0.49 | 0.31 | 0.29 |
| HAP(001) + water + CO2 (169), previous build | | 2.70 | 1.35 | 0.83 | 0.77 |
| HAP(001) + water + CO2 (169) | 15.7 | 1.38 | 0.84 | 0.45 | 0.42 |
| monetite (224), previous build | | 3.10 | 1.26 | 0.86 | 0.75 |
| monetite (224) | 14.6 | 1.64 | 0.88 | 0.53 | 0.48 |

(The previous build's monetite row is the morning's record: its rerun
overlapped another load on the card.)

**ms per evaluation**, water (`omnimd-gpu-rtx4090-2026-11-10-r3.csv`):

| atoms | fp64 | fp64 compressed | fp32 | fp32 compressed |
|---:|---:|---:|---:|---:|
| 192, previous build | 1.97 | 1.04 | 0.69 | 0.65 |
| 192 | 1.47 | 0.83 | 0.47 | 0.39 |
| 1,536, previous build | 13.1 | 6.28 | 2.43 | 1.92 |
| 1,536 | 9.21 | 5.10 | 1.53 | 1.22 |
| 12,288, previous build | 99.6 | 46.1 | 16.7 | 12.4 |
| 12,288 | 70.1 | 38.7 | 9.59 | 7.24 |
| 98,304, previous build | 798 | 367 | 132 | 100 |
| 98,304 | 564 | 304 | 76.6 | 57.4 |

- Molecular dynamics of the HAP systems is 1.4 to 2.1 times faster per step
  than a day earlier, and 9 to 17 times (float64) or 30 to 47 times
  (compressed float32) faster than the LAMMPS record. Large systems gain 1.2
  to 1.7 times.
- Agreement is unchanged. All 83 HAP test frames, four models, four modes
  against DeePMD (`hap_check.py`): float64 within 1.8e-13 eV per atom and
  5.6e-13 eV/Å, float32 within 2.4e-7 and 7.5e-6, compressed float64 within
  8.4e-10 and 3.9e-6. Water against DeePMD float64: energy identical, forces
  within 1.8e-17 eV/Å (8.5e-16 compressed). The four-model deviation matches
  the one from DeePMD's forces as before.
- What changed, in the order of what it gave for 88 atoms (compressed
  float32, 0.53 ms per evaluation before):
  - One upload, one download and kept buffers per evaluation instead of
    about thirty device calls: 0.38 ms. The neighbor search alone had made
    fifteen.
  - The neighbor search ranks candidates without running its tie-breaking
    arithmetic for every pair (the compiler had made the branch
    unconditional), scans in single precision before measuring candidates in
    double, and reads positions in cell order: 0.33 ms.
  - No divisions or square roots in double precision per neighbor slot, and
    no work for empty slots (65% of the slots in these systems): 0.265 ms.
- Where the time goes now, 88 atoms, compressed float32 (extra time of
  running one stage twice): fitting nets 0.11 ms, forces 0.03, neighbor
  search 0.03, descriptor 0.025, embedding tables 0.015, environment 0.01.
  The fitting kernel is bound by loading its weights from global memory
  (about 20 ns per weight line and warp); see the section above for what
  was done about it.
  At 98,304 atoms (57 ms): forces 16.3 ms, descriptor 12.6, embedding tables
  10.0, fitting 8.6, neighbor search 5.6.
- The trainer shares the reworked environment, descriptor and force kernels:
  2 to 4% faster in float32, unchanged in float64 (numbers in the section
  above). Its time is in the fitting passes.

# Training on the HAP iteration-0 data, 2026-11-10

`omnimd-native-gpu-train` against the HAP pipeline's four iteration-0 DeePMD
models, on the frames those were trained on (171 frames in 8 systems: the
pipeline's iteration-0 training directory without the 25 lime frames that
were labelled after that training had started), with the same settings
(se_e2_a, `sel` [44, 16, 60, 72, 20], embedding [25, 50, 100], fitting
[240, 240, 240], 2,664,455 parameters, learning rate 1e-3 to 3e-8, 300,000
steps of one frame) and the same loss (energy and force terms; the virial
prefactors 0 to 0.02 leave that term off in DeePMD-kit's PyTorch backend and
in omniMD). Four omniMD models with seeds 11 to 14, trained at once with
float32 networks over float64 parameters; DeePMD trained in float64. Scored
on the pipeline's 83-frame sealed test with `omnimd-native-gpu-test`.

| | omniMD, four models | DeePMD-kit 3.1.3, four models |
|---|---:|---:|
| time for the four | 30 min at once | 30.8 h one after the other |
| sealed test, force RMSE (eV/Å) | 0.0698 to 0.0723 | 0.0699 to 0.0711 |
| sealed test, energy RMSE (meV/atom) | 1.64 to 1.78 | 1.64 to 1.74 |

- Same quality: the ranges of the two sets of four seeds overlap in every
  system, lime included (per-system table in
  [`native/GPU_TRAINING.md`](../../native/GPU_TRAINING.md)). omniMD's seeds
  spread a little wider in the overall force RMSE (3.6 against 1.8 percent).
- About sixty times less time for the ensemble. DeePMD's is the pipeline's
  record on this machine (file times of the four runs; 12.3 steps/s for one
  model alone, 6.5 each for two at once), not re-run here.
- Same exploration. The pipeline's sixteen iteration-0 exploration runs
  (20,000 steps each, a frame every 50) were repeated with omniMD's models
  and, as a control, with DeePMD's own models converted, both through
  `omnimd-mlmd` (`hap_explore.py`). In the eleven runs where the pipeline's
  LAMMPS record has most frames below the 0.1 eV/Å edge of the window, the
  fraction below it differs from the record by at most 0.09 with omniMD's
  models (0.03 without monetite at 300 K, where the edge lies in the middle
  of the distribution) and by at most 0.01 with DeePMD's models; the mean
  deviation differs by -6 to +25 percent and by -14 to +9 percent. In the
  other five runs (the HAP(001) interfaces, monetite at 500 K) the models
  fail in all three. The sixteen runs with their four-model deviations took
  5.1 minutes, 0.58 to 1.21 ms per MD step.
- The same instrument: `omnimd-native-gpu-test` gives the pipeline's recorded
  sealed-test errors of the converted DeePMD models to every printed digit
  (`hap-test-deepmd-iter0-by-omnimd-rtx4090-2026-11-10.json`).
- Which frames DeePMD saw is reconstructed from the pipeline's notes, the
  times of the label files (five lime frames predate the training) and the
  missing lime validation system; that the other systems' frames are
  unchanged since then is an assumption.

Raw numbers: `hap-train-ensemble-float32-omnimd-rtx4090-2026-11-10.json`
(sealed test), `hap-explore-iter0-omnimd-float32-rtx4090-2026-11-10.json` and
`hap-explore-iter0-deepmd-models-by-omnimd-rtx4090-2026-11-10.json`
(exploration).

**Runs on other frames or with another loss, kept as records.** They are
not comparisons with the pipeline's models:

| files (`hap-train-ensemble-float32-…`, `hap-explore-iter0-omnimd-float32-…`) | frames | loss | parameters | force RMSE | energy RMSE | lime force RMSE |
|---|---|---|---|---:|---:|---:|
| `with-lime-virial` | 196, all 30 lime | virial ramped in | float32 | 0.0711 to 0.0742 | 1.47 to 1.58 | 0.0125 to 0.0140 |
| `with-lime` | 196, all 30 lime | energy, force | float32 | 0.0712 to 0.0735 | 1.76 to 1.84 | 0.0129 to 0.0144 |
| `without-lime` | 166, no lime | energy, force | float64 | 0.0771 to 0.0938 | 242 to 763 | 0.1256 to 0.2307 |
| (the comparison above) | 171, 5 lime | energy, force | float64 | 0.0698 to 0.0723 | 1.64 to 1.78 | 0.0183 to 0.0252 |

- The pipeline's models score 0.020 to 0.021 on lime: between the runs with
  thirty lime frames and with none, which is how the difference in the
  training sets was found. Without lime frames the other systems are scored
  like the pipeline's; only lime fails, and with it the overall numbers.
- With float32 parameters the energy error is higher than with float64
  parameters under the same float32 networks: steps at the end of the
  schedule are below the spacing of float32 numbers. One member, last third
  repeated: 1.84 meV/atom with float32 parameters, 1.61 with float64
  networks, 1.61 with float32 networks over float64 parameters.
- `hap-train-omnimd-rtx4090-2026-11-10.json` is one float64 model of the
  first version of the trainer on the 196 frames with the virial ramped in,
  in 88 minutes (force RMSE 0.0719, energy 1.27 meV/atom).

**Trainer speed** on this model and the 196 frames (steps per second; 20,000
steps in float32, 6,000 in float64, nothing else on the card):

| requests at once | float32, each | float32, total | float64, each | float64, total |
|---:|---:|---:|---:|---:|
| 1, first version | 118 | 118 | 58 | 58 |
| 1 | 604 | 604 | 289 | 289 |
| 2 | 324 | 648 | 160 | 321 |
| 4 | 179 | 715 | 88 | 352 |

- From the first version to this one: networks evaluated as sets (284 and 92
  for one request), parameter gradients in one kernel (402 and 191), no
  embedding rows for empty neighbor slots, of which these frames have about
  two thirds (567 and 297), neighbor lists kept per frame (661 and 294),
  float64 parameters under float32 networks and float32 dot products in
  partial sums (604 and 289).
- A float32 step takes 1.87 ms with every stage timed, of which the two
  fitting passes take 0.85; requests trained at once gain 7 to 22 percent in
  total. Separate trainer processes did not add up with the first version
  (two at once 24.8 steps/s each, four 12.3 each, against 57.9 for one).
- The training set can be produced by omniMD alone: `import-deepmd` for each
  system with `id_prefix` and the common `type_map`, then `merge-datasets`,
  give the 196 frames of the directory in the order of the preparation
  script, with no numeric difference.

# Constant pressure against the pipeline's LAMMPS run, 2026-11-10

HAP bulk (88 atoms), 300 K, 1.01325 bar, fully flexible cell, 0.5 fs, 100,000
steps, sampled every 50 steps, driven by the iteration-4 model 0. LAMMPS: the
pipeline's own exploration run (`fix npt temp 300 300 0.05 tri 1.01325 1.01325
0.5`, Nose-Hoover chains; model 0 drives and all four TensorFlow models are
evaluated every 50 steps), read from its `log.lammps` and `traj.dump`.
omniMD: `npt_csvr_crescale` with the same temperature, pressure and time
constants (CSVR thermostat 50 fs,
stochastic cell rescaling 500 fs, assumed compressibility 1e-6 per bar,
`triclinic` coupling, a cell move every 10 steps), started from the same
`conf.lmp` (`hap_npt.py`, `hap-npt-omnimd-rtx4090-2026-11-10-*.json`). The
algorithms differ, so the trajectories do; means over the last 40 ps with
standard errors from eight block averages:

| quantity | LAMMPS (recorded) | omniMD float64 | omniMD float64-compressed | omniMD float32 | omniMD float32-compressed |
|---|---:|---:|---:|---:|---:|
| volume (Å³) | 1080.18 ± 0.13 | 1080.82 ± 1.07 | 1080.57 ± 1.11 | 1080.66 ± 1.11 | 1080.70 ± 1.00 |
| potential energy (eV) | -45045.160 ± 0.007 | -45045.174 ± 0.028 | -45045.167 ± 0.025 | -45045.172 ± 0.023 | -45045.171 ± 0.026 |
| temperature (K) | 302.0 ± 0.9 | 300.1 ± 1.5 | 300.0 ± 1.7 | 300.2 ± 1.8 | 300.4 ± 1.5 |
| a (Å) | 9.469 ± 0.000 | 9.485 ± 0.012 | 9.484 ± 0.012 | 9.484 ± 0.012 | 9.485 ± 0.012 |
| b (Å) | 9.471 ± 0.001 | 9.467 ± 0.009 | 9.466 ± 0.009 | 9.467 ± 0.009 | 9.467 ± 0.009 |
| c (Å) | 13.909 ± 0.001 | 13.922 ± 0.019 | 13.921 ± 0.019 | 13.922 ± 0.019 | 13.922 ± 0.019 |
| alpha (°) | 89.97 ± 0.02 | 90.02 ± 0.16 | 90.04 ± 0.15 | 90.05 ± 0.15 | 90.03 ± 0.15 |
| beta (°) | 90.00 ± 0.02 | 89.87 ± 0.11 | 89.86 ± 0.11 | 89.83 ± 0.12 | 89.85 ± 0.11 |
| gamma (°) | 119.99 ± 0.00 | 120.14 ± 0.12 | 120.15 ± 0.11 | 120.15 ± 0.12 | 120.16 ± 0.12 |
| volume std (Å³) | 7.00 | 7.10 | 7.16 | 7.17 | 7.09 |
| pressure (bar) | -1 ± 12 | -607 ± 753 | -585 ± 753 | -612 ± 772 | -591 ± 762 |
| refused cell moves | | 10 of 10000 | 9 of 10000 | 8 of 10000 | 8 of 10000 |
| ms per step | 45.9 | 1.68 | 0.90 | 0.60 | 0.56 |

- Every mean of every mode agrees with the LAMMPS run within 1.4 combined
  standard errors, and the volume fluctuation is the same (7.0 to 7.2 against
  7.0 Å³). The omniMD errors are larger because a first-order barostat lets
  the volume diffuse where Nose-Hoover makes it oscillate: over a time t the
  mean pressure of such a barostat is uncertain by about
  sqrt(2 kT tau / (compressibility V t)), here 1000 bar for 40 ps. The mean
  pressure is therefore a weak check on this time scale and the volume the
  meaningful one.
- About one cell move in a thousand is refused by the acceptance test.
- Each cell move costs one extra evaluation, so NPT with a move every 10
  steps costs 10 to 15 percent more per step than NVT (float64 1.68 against
  1.50 ms; float32 compressed 0.56 against 0.49 ms).
- The recorded LAMMPS time, 45.9 ms per step, comes from the pipeline's
  exploration batch, in which several runs shared the card, so it is not a
  like-for-like speed comparison; alone, LAMMPS takes 13.6 ms per NVT step for
  this system (table below).
- Two runs of the same request do not repeat bit for bit on the GPU runtime
  (the order of atomic force additions varies): a second float64 run gave
  1080.76 ± 1.02 Å³. The four modes above share the seed of thermostat and
  barostat, so their volumes move together; three further seeds of 50 ps
  (float32 compressed) gave 1081.65 ± 0.82, 1080.79 ± 1.43 and 1080.85 ±
  0.87 Å³, and a 400 ps run with a fifth seed 1080.08 ± 0.28 Å³ (LAMMPS:
  1080.18 ± 0.13) with a mean pressure of -17 ± 197 bar, potential energy
  -45045.167 ± 0.004 eV, a = 9.462 ± 0.006, b = 9.474 ± 0.007, c = 13.919 ±
  0.008 Å and gamma = 120.02 ± 0.04 degrees
  (`hap-npt-omnimd-rtx4090-2026-11-10-float32-compressed-400ps.json`).
- The effective energy (total minus thermostat heat minus barostat work)
  drifts by at most 0.009 eV over the 50 ps in every mode.
- The exploration chain on the same system (`hap_npt.py --deviation`,
  `hap-explore-omnimd-rtx4090-2026-11-10.json`): the 2,001 sampled frames of
  a float64 run went through `import-trajectory` and `model-deviation` with
  the four models on the GPU runtime. The mean over frames of the largest
  force deviation is 0.0338 ± 0.0002 eV/Å (the LAMMPS run's `model_devi.out`:
  0.0333 ± 0.0002) and of the mean force deviation 0.01603 ± 0.00004
  (0.01595 ± 0.00005). With the pipeline's window of 0.1 to 0.3 eV/Å both
  count all 2,001 frames as accurate (largest value 0.078 against 0.066).
  Import and deviation of the 2,001 frames took 14.5 s.

# HAP pipeline ensemble: agreement and exploration-style speed, 2026-11-10

Models and frames from the HAP CO2-mineralization pipeline v2 on this machine
(iteration 4: four se_e2_a models, type map Ca P O H C, sel [44, 16, 60, 72, 20],
embedding [25, 50, 100], fitting [240, 240, 240], float64; sealed test of 83
DFT-labelled frames in 8 systems, 54 to 224 atoms). The pipeline marks these
models as not production-qualified; the numbers below are agreement and speed
only. Prepared with `hap_prepare.py`, which reads the pipeline tree and writes
only to `cases/hap/`.

**Agreement with DeePMD-kit 3.1.3** (`hap_check.py`; DeepPot on CPU, all 83
frames, largest difference over the four models):

| omniMD mode | energy per atom (eV) | force (eV/Å) | virial (eV) |
|---|---:|---:|---:|
| float64 | 1.8e-13 | 5.6e-13 | 1.1e-12 |
| float64, compressed | 8.4e-10 | 3.9e-6 | 2.0e-5 |
| float32 | 2.4e-7 | 7.5e-6 | 7.9e-5 |
| float32, compressed | 2.8e-7 | 8.7e-6 | 7.2e-5 |

- The sealed-test errors against the DFT labels come out the same from omniMD
  as from DeePMD (largest shift 6e-14 in float64, 2.4e-7 in float32
  compressed, in eV/atom or eV/Å), and both reproduce the pipeline's recorded
  `sealed_test.json` to its printed precision (3.6e-7 relative).
- No frame exceeds `sel`, and no frame leaves the table range at
  `--compress 0.5`.
- Compression costs more accuracy here than on the untrained water model
  (4e-6 against 9e-16 eV/Å) because trained embedding nets vary faster. It
  remains four orders of magnitude below the models' own test error
  (0.014 to 0.098 eV/Å per system and model).
- Model deviation (`hap_md.py`): `omnimd-mlmd model-deviation` with the four
  models on the GPU runtime gives max/min/mean force deviations within 1e-14
  eV/Å of those computed from DeePMD's reference forces. The whole run over
  83 frames takes about 2 s, including loading four models.

**Speed, ms per MD step** (NVT at 300 K, 0.5 fs). DeePMD: LAMMPS
`pair_style deepmd` with the pipeline's protocol and models (TensorFlow
graphs, skin 0 with a rebuild every step, 2 OpenMP threads;
`hap-lammps-rtx4090-2026-11-10.csv`). omniMD: `omnimd-mlmd run` on the GPU
runtime with the CSVR thermostat, sampled every 50 steps
(`hap-omnimd-rtx4090-2026-11-10.csv`):

| system (atoms) | DeePMD 1 model | DeePMD 4 models | omniMD fp64 | fp64 compressed | fp32 | fp32 compressed |
|---|---:|---:|---:|---:|---:|---:|
| HAP bulk (88) | 13.6 | 12.8 | 1.50 | 0.78 | 0.52 | 0.49 |
| HAP(001) + water + CO2 (169) | 15.7 | 16.5 | 2.57 | 1.19 | 0.80 | 0.69 |
| monetite (224) | 14.6 | 15.5 | 3.10 | 1.26 | 0.86 | 0.75 |

- omniMD float64 gives DeePMD's forces to rounding and is 4.7 to 9 times
  faster per step; compressed float64 is 11 to 17 times faster and compressed
  float32 19 to 28 times.
- "4 models" is the pipeline's exploration setting: model 0 drives the
  dynamics and all four are evaluated every 50 steps. omniMD runs the
  deviation as a separate command over the sampled frames (about 25 ms per
  frame including model loading, so under 1 ms per MD step at this sampling).
  Since 2026-11-11 a run can carry the members along; see the top of this
  file.
- The pipeline's own 100,000-step exploration logs show 52 ms per step for the
  169-atom system (three jobs sharing the GPU, 87 min per run).
- The thermostats differ (Nose-Hoover in LAMMPS, CSVR in omniMD); this does
  not affect the cost per step. The pipeline's 300 K bulk runs are NPT; see
  the constant-pressure comparison above.
- These systems are bound by per-step latency, not arithmetic. What made
  them fast here: one launch for all 25 embedding nets and one fused
  forward-and-backward kernel for all five fitting nets (169 atoms, float32
  compressed: 1.41 to 0.67 ms per evaluation).

# Water benchmark rerun with the single-launch kernels, 2026-11-10 (r2)

`omnimd-gpu-rtx4090-2026-11-10-r2.csv`, same protocol as the table below.

| atoms | fp64 | fp64 compressed | fp32 | fp32 compressed |
|---:|---:|---:|---:|---:|
| 192 | 2.07 | 1.05 | 0.70 | 0.63 |
| 1,536 | 12.4 | 5.88 | 2.38 | 1.81 |
| 12,288 | 95.7 | 44.2 | 16.0 | 11.9 |
| 98,304 | 763 | 352 | 128 | 94.8 |

The machine ran about 10% slower during this run than during the morning run
below: the morning build, rebuilt and alternated with the new one, also takes
11.6 ms at 12,288 atoms in float32 compressed (10.4 in the morning). At equal
conditions the new build is faster at 192 atoms (float64 2.77 to 2.07 ms,
compressed 1.65 to 1.05) and within 1.5% elsewhere. Agreement with the DeePMD
reference is unchanged.

# Fused embedding, compression and DeePMD compressed baseline, 2026-11-10

Same card, model and frames as below; ms per energy/force/virial evaluation.
DeePMD rows are LAMMPS `pair_style deepmd` (uncompressed from 2026-11-09,
compressed from `deepmd-3.1.3-lammps-compressed-2026-11-10.csv`, models from
`compress_deepmd.py`). omniMD rows are `omnimd-gpu-rtx4090-2026-11-10.csv`:
10 to 100 in-process evaluations each; "compressed" is `--compress 0.5`
(piecewise quintic embedding tables, step 0.01).

| atoms | precision | DeePMD | omniMD | speed-up | DeePMD compressed | omniMD compressed | speed-up |
|---:|---|---:|---:|---:|---:|---:|---:|
| 192 | fp32 | 10.4 | 0.70 | 14.9× | 7.58 | 0.69 | 11.0× |
| 1,536 | fp32 | 13.1 | 2.18 | 6.0× | 8.94 | 1.64 | 5.5× |
| 12,288 | fp32 | 66.1 | 14.2 | 4.7× | 32.5 | 10.4 | 3.1× |
| 98,304 | fp32 | error | 118 | — | error | 84.9 | — |
| 192 | fp64 | 11.4 | 2.77 | 4.1× | 8.57 | 1.65 | 5.2× |
| 1,536 | fp64 | 41.9 | 11.8 | 3.5× | 20.2 | 6.12 | 3.3× |
| 12,288 | fp64 | 293 | 90.4 | 3.2× | 118 | 40.0 | 2.9× |
| 98,304 | fp64 | error | 699 | — | error | 329 | — |

- Changes since 2026-11-09:
  - The embedding net runs as one fused kernel (12,288 atoms fp32: embedding
    stage 11.0 → 5.3 ms).
  - Optional compression replaces it by tables (5.3 → 1.5 ms; fp64 53 → 7 ms).
  - Neighbor selection follows DeePMD-kit 3.
- Accuracy against DeepMD-kit CPU float64 `DeepPot` (192 and 1,536 atoms):
  omniMD fp64 energy is identical with and without compression; the largest
  force difference is 2e-17 eV/Å exact and 9e-16 eV/Å compressed. fp32 forces
  are within 6e-9 eV/Å in both modes. DeePMD's own compressed accuracy was not
  measured.
- Remaining cost at 12,288 atoms, fp32 compressed: force/virial kernel 4.0 ms,
  descriptor 1.9, embedding tables 1.5, neighbor list 1.3, fitting 0.9.
- LAMMPS rebuilds its neighbor list every ~13 steps; omniMD rebuilds on every
  evaluation. At 98,304 atoms LAMMPS stops with a TorchScript runtime error in
  every mode (probably out of memory; not verified).

# GPU se_e2_a and LAMMPS baseline, 2026-11-09 (same card, model and frames)

ms per energy/force/virial evaluation. DeePMD LAMMPS rows are `pair_style deepmd`
NVE loop time per step (`deepmd-3.1.3-lammps-2026-11-09.csv`); omniMD rows are the
in-process GPU evaluator, `omnimd-native-gpu-eval --precision P --bench K`
(`omnimd-gpu-rtx4090-2026-11-09.csv`, stage times and errors in the `.json`).
omniMD float32 runs the networks in float32; geometry, the B accumulation,
forces and virial stay float64.

| atoms | DeePMD LAMMPS fp64 | omniMD GPU fp64 | speed-up | DeePMD LAMMPS fp32 | omniMD GPU fp32 | speed-up |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 11.4 | 3.52 | 3.2× | 10.4 | 1.01 | 10.3× |
| 1,536 | 41.9 | 15.7 | 2.7× | 13.1 | 2.86 | 4.6× |
| 12,288 | 293 | 117 | 2.5× | 66.1 | 20.4 | 3.2× |
| 98,304 | error | 929 | — | error | 175 | — |

- Accuracy against DeepMD-kit CPU float64 `DeepPot` (192 and 1,536 atoms): omniMD
  float64 energy is identical and the largest force difference is 2e-17 eV/Å. In
  float32 the force error is at most 5.4e-9 eV/Å. At 12,288 and 98,304 atoms,
  float32 is within 8.8e-9 eV/Å of omniMD float64.
- LAMMPS rebuilds its neighbor list only every ~13 steps (2 Å skin), while omniMD
  rebuilds on every evaluation. The comparison therefore slightly favors DeePMD.
- At 98,304 atoms LAMMPS stopped with a TorchScript runtime error in both
  precisions. This is probably out of memory, but that has not been verified.
- Milestone M2 (12k atoms fp32 ≥ 1.0× DeePMD) is met: 3.2×.

# W1 baseline, 2026-10-29 (RTX 4090 24 GB, Windows 11 host)

Water `se_e2_a` (sel [46, 92], embedding [25, 50, 100], fitting [240, 240, 240]),
same parameters in both engines. Raw rows: `deepmd-3.1.3-rtx4090-2026-10-29.csv`,
`omnimd-rtx4090-2026-10-29.csv`. Times are ms per energy/force/virial evaluation.

| atoms | DeePMD GPU fp64 | DeePMD GPU fp32 | DeePMD CPU fp64 (8 thr) | DeePMD CPU fp32 (8 thr) | omniMD CPU fp64 (1 thr) | omniMD CUDA fp64 |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 39.9 | 36.7 | 128 | 79.5 | 299 | rejected |
| 1,536 | 68.8 | 39.0 | 1,111 | 691 | 4,599 | rejected |
| 12,288 | OOM | not run | n/a | n/a | n/a | rejected |
| 98,304 | OOM | not run | n/a | n/a | n/a | rejected |

DeePMD is the Python `DeepPot` route (PyTorch backend, DeepMD-kit 3.1.3,
`ghcr.io/deepmodeling/deepmd-kit:3.1.3_cuda129`). GPU peak allocator memory was
128 MB at 192 atoms and 1.97 GB at 1,536 atoms. CPU cases above 1,536 atoms were
not run by design.

## Findings

1. **omniMD CUDA cannot run a production-size model.** The CUDA subset requires
   embedding width ≤ 64 and descriptor width ≤ 128; this model has 100 and
   1,600. There is no GPU comparison point yet. Plan item I-3 (batched GEMM, no
   width limits) in W5–W7 has to close this gap before M2.
2. **omniMD CPU is 2.3× (192 atoms) to 4.1× (1,536 atoms) slower than DeePMD
   CPU** in float64. omniMD runs single-threaded against DeePMD's 8 threads, so the
   I-7 target (≤ DeePMD CPU at equal threads) looks reachable at 192 atoms. The
   ratio grows with size, which fits the brute-force neighbor search (I-1).
3. **DeePMD's Python route stops at 12,288 atoms.** `DeepPot` ran out of GPU memory
   even at batch size 1. The large-system baseline must come from LAMMPS
   `pair_style deepmd`, which uses its own neighbor list.

## Not measured yet

- **LAMMPS `pair_style deepmd`:** every case failed in this run because the image
  runs as root and Open MPI refuses that. The script also handed LAMMPS a relative
  log path. Both are fixed; see the 2026-11-09 table above.
- **DeePMD fp32 at 12,288 and 98,304 atoms:** the Docker Desktop VM went down
  during the fp32 12,288-atom DeepPot case and the run ended there.
- **Compressed DeePMD models** (`dp compress`), which the W1 row of the plan lists,
  were not built yet.
- **Rerun command once Docker is back:**
  `bench_deepmd.py cases results/deepmd-3.1.3-lammps-<tag>.csv --skip-deeppot`.

omniMD timings come from the CLI. They subtract the fastest one-frame run from
the fastest K-frame run (3 repeats) and include per-frame JSON parsing.
