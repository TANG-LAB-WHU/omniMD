#!/usr/bin/env python3
"""One explicit native-CUDA CLI NVE parity/restart replay; no C API or timing."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def difference(a, b):
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), "shape mismatch"
        return max((difference(x, y) for x, y in zip(a, b)), default=0.0)
    assert isinstance(a, (int, float)) and isinstance(b, (int, float))
    assert math.isfinite(a) and math.isfinite(b), "nonfinite result"
    return abs(a - b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--run-gpu", action="store_true",
                        help="explicitly permit the two native CUDA runs")
    args = parser.parse_args()
    if not args.run_gpu:
        parser.error("--run-gpu is required; coordinate an exclusive GPU window first")
    root = Path(__file__).resolve().parents[2]
    binary = args.binary.resolve(strict=True)
    model = root / "mlmd/native/fixtures/full-tanh-v1.json"
    frame = json.loads((root / "mlmd/native/fixtures/frame.jsonl").read_text())
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    common = {
        "dynamics_units": "EvAngstromFsU", "frame": frame,
        "masses": [15.999, 1.008, 1.008, 15.999],
        "velocities": [[0.001, 0.002, -0.001], [-0.002, 0.001, 0.003],
                       [0.001, -0.003, 0.002], [-0.001, 0.001, -0.002]],
        "ensemble": {"kind": "nve"}, "timestep_fs": 0.001,
        "steps": 8, "sample_every": 1, "checkpoint_every": 4,
    }

    def run(name, request, backend, start, steps):
        request_path = work / (name + ".request.json")
        output = work / (name + ".jsonl")
        request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf8")
        # Sequential launches; nonzero completion raises before any subsequent run.
        # No timeout-driven retries or device reset are performed.
        with (work / (name + ".log")).open("w", encoding="utf8") as log:
            proc = subprocess.run([str(binary), "run", str(request_path), str(output)],
                                  stdout=log, stderr=subprocess.STDOUT)
        assert proc.returncode == 0, f"{name} failed ({proc.returncode}); see {name}.log; stop"
        records = [json.loads(line) for line in output.read_text().splitlines()]
        header, completed = records[0], records[-1]
        assert header["kind"] == "header" and completed["kind"] == "completed"
        assert header["metadata"]["backend"] == backend
        assert header["model_sha256"] == sha256(model)
        assert header["dynamics_units"] == "EvAngstromFsU"
        assert header["start_step"] == start and header["requested_steps"] == steps
        assert completed["final_step"] == start + steps
        snapshots = [r for r in records if r["kind"] == "frame"]
        assert [r["step"] for r in snapshots] == list(range(start, start + steps + 1))
        assert all(r["frame"]["units"] == "EvAngstrom"
                   and r["evaluation"]["units"] == "EvAngstrom" for r in snapshots)
        return output, records, snapshots

    cpu_options = {"native_model": str(model), "device": "cpu"}
    gpu_options = {"native_model": str(model), "device": "cuda:0"}
    _, cpu_records, cpu = run("cpu", dict(common, options=cpu_options),
                              "native-se-e2-a-v1", 0, 8)
    gpu_output, gpu_records, gpu = run("cuda", dict(common, options=gpu_options),
                                      "native-cuda-se-e2-a-v1", 0, 8)
    maxima = {"energy_ev": 0.0, "forces_ev_per_angstrom": 0.0,
              "virial_ev": 0.0, "positions_angstrom": 0.0,
              "velocities_angstrom_per_fs": 0.0}
    for c, g in zip(cpu, gpu):
        assert c["step"] == g["step"] and c["time_fs"] == g["time_fs"]
        assert c["frame"]["species"] == g["frame"]["species"]
        assert c["frame"]["cell"] == g["frame"]["cell"]
        values = {
            "energy_ev": difference(c["evaluation"]["energy"], g["evaluation"]["energy"]),
            "forces_ev_per_angstrom": difference(c["evaluation"]["forces"], g["evaluation"]["forces"]),
            "virial_ev": difference(c["evaluation"]["virial"], g["evaluation"]["virial"]),
            "positions_angstrom": difference(c["frame"]["positions"], g["frame"]["positions"]),
            "velocities_angstrom_per_fs": difference(c["velocities"], g["velocities"]),
        }
        for key, value in values.items():
            maxima[key] = max(maxima[key], value)
    tolerances = {key: (1e-10 if key.startswith(("positions", "velocities")) else 1e-9)
                  for key in maxima}
    assert all(maxima[key] <= tolerances[key] for key in maxima), maxima
    checkpoint = Path(str(gpu_output) + ".step-4.checkpoint.json")
    assert checkpoint.is_file(), "periodic checkpoint missing"
    resumed_path, _, resumed = run("cuda-resumed", {
        "options": gpu_options, "dynamics_units": "EvAngstromFsU",
        "restart": str(checkpoint), "timestep_fs": common["timestep_fs"],
        "steps": 4, "sample_every": 1,
    }, "native-cuda-se-e2-a-v1", 4, 4)
    assert gpu[4:] == resumed, "resumed snapshots differ"
    final_cp = json.loads(Path(str(gpu_output) + ".checkpoint.json").read_text())
    resumed_cp = json.loads(Path(str(resumed_path) + ".checkpoint.json").read_text())
    assert final_cp == resumed_cp, "final checkpoint differs"
    report = {
        "schema": "omnimd-native-cuda-cli-acceptance-v1", "status": "passed",
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model": "mlmd/native/fixtures/full-tanh-v1.json", "model_sha256": sha256(model),
        "binary_sha256": sha256(binary), "script_sha256": sha256(Path(__file__)),
        "backend_cpu": cpu_records[0]["metadata"]["backend"],
        "backend_cuda": gpu_records[0]["metadata"]["backend"],
        "ensemble": "nve", "atoms": 4, "cell": "triclinic", "steps": 8,
        "timestep_fs": common["timestep_fs"], "snapshots_compared": len(cpu),
        "max_absolute_difference": maxima, "absolute_tolerances": tolerances,
        "periodic_checkpoint_step": 4, "resumed_steps": 4,
        "identical_resumed_snapshots": True, "identical_final_checkpoint": True,
        "cpu_max_energy_drift_ev": cpu_records[-1]["max_absolute_energy_drift_ev"],
        "cuda_max_energy_drift_ev": gpu_records[-1]["max_absolute_energy_drift_ev"],
        "scope": "Short unified native CLI routing and restart acceptance; not a stability or speed benchmark. C API GPU was not invoked.",
    }
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
