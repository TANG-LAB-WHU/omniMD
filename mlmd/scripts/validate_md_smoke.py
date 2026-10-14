#!/usr/bin/env python3
"""Check the MD CLI's native path and restart a previously validated C-API run."""
import argparse
import json
from pathlib import Path
import subprocess

def records(path):
    data = [json.loads(line) for line in path.read_text().splitlines()]
    assert data[-1]["kind"] == "completed"
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--native-model", type=Path, required=True)
    parser.add_argument("--native-frame", type=Path, required=True)
    parser.add_argument("--reference-request", type=Path, required=True)
    parser.add_argument("--reference-trajectory", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)

    def run(name, request):
        input_path = args.work / (name + ".request.json")
        output_path = args.work / (name + ".jsonl")
        input_path.write_text(json.dumps(request), encoding="utf8")
        proc = subprocess.run([str(args.binary), str(input_path), str(output_path)],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (args.work / (name + ".log")).write_text(proc.stdout, encoding="utf8")
        assert proc.returncode == 0, proc.stdout
        return output_path, records(output_path)

    native_frame = json.loads(args.native_frame.read_text().strip())
    masses = [{"O": 15.999, "H": 1.008}[s] for s in native_frame["species"]]
    _, native = run("native", {
        "options": {"native_model": str(args.native_model.resolve())},
        "frame": native_frame, "masses": masses, "velocities": [[0., 0., 0.] for _ in masses],
        "timestep_fs": 0.01, "steps": 20, "sample_every": 10,
    })
    reference = records(args.reference_trajectory)
    original = json.loads(args.reference_request.read_text())
    assert original["steps"] >= 2
    first = dict(original, steps=original["steps"] // 2)
    first_path, _ = run("first", first)
    second = {key: original[key] for key in ("options", "timestep_fs", "sample_every")}
    second.update(restart=str(first_path) + ".checkpoint.json",
                  steps=original["steps"]-first["steps"])
    _, resumed = run("resumed", second)
    expected = next(item for item in reversed(reference) if item["kind"] == "frame")
    actual = next(item for item in reversed(resumed) if item["kind"] == "frame")
    def error(a, b):
        if isinstance(a, list):
            assert len(a) == len(b)
            return max(error(x, y) for x, y in zip(a, b))
        return abs(a-b)
    differences = {
        "position_angstrom": error(actual["frame"]["positions"], expected["frame"]["positions"]),
        "velocity_angstrom_per_fs": error(actual["velocities"], expected["velocities"]),
        "energy_ev": error(actual["evaluation"]["energy"], expected["evaluation"]["energy"]),
        "forces_ev_per_angstrom": error(actual["evaluation"]["forces"], expected["evaluation"]["forces"]),
        "virial_ev": error(actual["evaluation"]["virial"], expected["evaluation"]["virial"]),
    }
    assert actual["step"] == expected["step"]
    assert max(differences.values()) < 1e-10, differences
    report = {"native_steps": native[-1]["final_step"],
              "native_max_energy_drift_ev": native[-1]["max_absolute_energy_drift_ev"],
              "resumed_step": actual["step"], "max_absolute_restart_difference": differences}
    (args.work/"report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
