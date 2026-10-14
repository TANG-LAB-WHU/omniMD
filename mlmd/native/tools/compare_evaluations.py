"""Compare two shared-contract Evaluation JSONL streams, using only the stdlib.

Inputs must come from the same ordered frames and weights; this cannot check
model equivalence or fix type order/units. Exit 0: within tolerances, 1: mismatch,
2: malformed input. No model runtime is required.
"""
import argparse
import json
import math
import sys
from pathlib import Path


def read(path):
    results = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or set(item) != {"energy", "forces", "virial", "units"}:
            raise ValueError(f"{path}:{number}: expected shared-contract Evaluation")
        if item["units"] != "EvAngstrom":
            raise ValueError(f"{path}:{number}: wrong units")
        if not isinstance(item["forces"], list) or not item["forces"]:
            raise ValueError(f"{path}:{number}: forces must be a nonempty [N,3] array")
        for field in ("forces", "virial"):
            rows = item[field]
            if not isinstance(rows, list) or (field == "virial" and len(rows) != 3):
                raise ValueError(f"{path}:{number}: invalid {field} shape")
            if any(not isinstance(row, list) or len(row) != 3 for row in rows):
                raise ValueError(f"{path}:{number}: invalid {field} row shape")
        values = [item["energy"], *sum(item["forces"], []), *sum(item["virial"], [])]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError(f"{path}:{number}: nonfinite or nonnumeric evaluation")
        results.append(item)
    if not results:
        raise ValueError(f"{path}: no evaluations")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--energy-atol", type=float, default=1e-8)
    parser.add_argument("--force-atol", type=float, default=1e-7)
    parser.add_argument("--virial-atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-7)
    args = parser.parse_args()
    tolerances = {"energy": args.energy_atol, "forces": args.force_atol, "virial": args.virial_atol}
    if any(not math.isfinite(v) or v < 0 for v in [args.rtol, *tolerances.values()]):
        parser.error("tolerances must be finite and nonnegative")
    try:
        reference, candidate = read(args.reference), read(args.candidate)
        if len(reference) != len(candidate):
            raise ValueError("frame count mismatch")
        maximum = dict.fromkeys(tolerances, 0.0)
        failures = 0
        worst = None
        for frame, (ref, got) in enumerate(zip(reference, candidate)):
            if len(ref["forces"]) != len(got["forces"]):
                raise ValueError(f"frame {frame}: atom count mismatch")
            for field, atol in tolerances.items():
                rv = [ref[field]] if field == "energy" else sum(ref[field], [])
                gv = [got[field]] if field == "energy" else sum(got[field], [])
                for component, (r, g) in enumerate(zip(rv, gv)):
                    error = abs(r-g)
                    maximum[field] = max(maximum[field], error)
                    if error > atol + args.rtol * abs(r):
                        failures += 1
                        if worst is None or error > worst["absolute_error"]:
                            worst = {"frame": frame, "field": field, "component": component,
                                     "reference": r, "candidate": g, "absolute_error": error}
        print(json.dumps({"frames": len(reference), "passed": failures == 0,
                          "failed_components": failures, "max_absolute_error": maximum,
                          "worst_mismatch": worst}, indent=2, allow_nan=False))
        return int(failures != 0)
    except (ValueError, TypeError, OSError, OverflowError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
