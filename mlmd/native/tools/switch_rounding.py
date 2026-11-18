"""Whether the switch of the GPU kernels can come out below zero just inside the cutoff.

    python switch_rounding.py

The kernels work the switch out as q = z^3 (1 + 3 t + 6 t^2) with
t = (r - rcut_smth) * inv_range, inv_range = 1 / (rcut - rcut_smth) and z = 1 - t,
all in float64. A neighbor has r < rcut, so z is not below zero in exact arithmetic;
in float64 it depends on where the rounding falls. The place of a neighbor in the
table of an embedding net of se_e2_a or se_e2_r is q / r over a width, and a place
below zero is outside the table: its frame would be evaluated with the nets.

For every pair of a cutoff and a smoothing radius in tenths of an angstrom, and the
last five float64 below the cutoff, this prints how many pairs give z < 0
- as the kernel is written: t is rounded, then taken from one;
- if a compiler fused the product into the subtraction: z = fl(1 - a * inv_range)
  with a = fl(r - rcut_smth) and the product not rounded.
"""
import math
from fractions import Fraction


def main():
    pairs = written = fused = 0
    examples = []
    for cutoff_tenths in range(20, 121):
        rcut = cutoff_tenths / 10.0
        for smooth_tenths in range(0, cutoff_tenths):
            rmin = smooth_tenths / 10.0
            inv_range = 1.0 / (rcut - rmin)
            pairs += 1
            below_written = below_fused = False
            r = rcut
            for step in range(1, 6):
                r = math.nextafter(r, -math.inf)
                a = r - rmin
                if 1.0 - a * inv_range < 0.0:
                    below_written = True
                exact = 1 - Fraction(a) * Fraction(inv_range)
                if exact < 0:
                    below_fused = True
                    if len(examples) < 4:
                        examples.append((rcut, rmin, step, float(exact)))
            written += below_written
            fused += below_fused
    print(f'{pairs} pairs of a cutoff (2.0 to 12.0) and a smoothing radius below it')
    print(f'z below zero for a neighbor inside the cutoff, as written: {written}')
    print(f'z below zero if the product were fused into the subtraction: {fused}')
    for rcut, rmin, step, z in examples:
        print(f'  rcut {rcut}, rcut_smth {rmin}, {step} float64 below the cutoff: z = {z:.1e}')


if __name__ == '__main__':
    main()
