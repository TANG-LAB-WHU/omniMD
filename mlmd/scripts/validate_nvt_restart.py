#!/usr/bin/env python3
"""Exercise seeded NVT CLI/check/periodic checkpoints and exact restart for either backend."""
import argparse
import json
from pathlib import Path
import subprocess

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    request = json.loads(a.request.read_text())
    assert request["ensemble"]["kind"] == "nvt_csvr" and request["steps"] >= 4
    request["checkpoint_every"] = request["steps"] // 4
    def run(name, r):
        ip = a.work / (name + ".request.json")
        op = a.work / (name + ".jsonl")
        ip.write_text(json.dumps(r), encoding="utf8")
        if name == "full":
            check = subprocess.run([str(a.binary), "check", str(ip)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (a.work/"check.log").write_text(check.stdout, encoding="utf8")
            assert check.returncode == 0, check.stdout
        proc = subprocess.run([str(a.binary), "run", str(ip), str(op)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (a.work/(name+".log")).write_text(proc.stdout, encoding="utf8")
        assert proc.returncode == 0, proc.stdout
        records = [json.loads(line) for line in op.read_text().splitlines()]
        assert records[-1]["kind"] == "completed"
        return op, records
    full_path, full = run("full", request)
    first_steps = request["steps"] // 2
    first_path, _ = run("first", dict(request, steps=first_steps))
    interval = first_path.with_name(first_path.name + f".step-{request['checkpoint_every']}.checkpoint.json")
    assert interval.exists()
    _, resumed = run("resumed", {
        "options": request["options"], "dynamics_units": "EvAngstromFsU",
        "restart": str(first_path) + ".checkpoint.json",
        "timestep_fs": request["timestep_fs"], "steps": request["steps"]-first_steps,
        "sample_every": request["sample_every"],
    })
    expected = next(r for r in reversed(full) if r["kind"] == "frame")
    actual = next(r for r in reversed(resumed) if r["kind"] == "frame")
    assert expected == actual, "resumed frame differs (including thermostat heat)"
    full_cp = json.loads(Path(str(full_path)+".checkpoint.json").read_text())
    resumed_cp = json.loads(Path(resumed[-1]["checkpoint"]).read_text())
    assert full_cp == resumed_cp, "checkpoint/RNG mismatch"
    zero = dict(request)
    zero.pop("initialize_velocities", None)
    zero["velocities"] = [[0., 0., 0.] for _ in zero["masses"]]
    ip = a.work/"zero-kinetic.request.json"
    ip.write_text(json.dumps(zero), encoding="utf8")
    rejected = subprocess.run([str(a.binary), "check", str(ip)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (a.work/"zero-kinetic.log").write_text(rejected.stdout, encoding="utf8")
    assert rejected.returncode != 0 and "nonzero initial kinetic" in rejected.stdout
    report = {"backend": full[0]["metadata"]["backend"], "steps": request["steps"],
        "timestep_fs":request["timestep_fs"], "model_sha256":full_cp["model_sha256"],
        "identical_final_frame_and_checkpoint":True, "periodic_checkpoint_verified":True,
        "zero_initial_kinetic_rejected":True, "degrees_of_freedom":full[0]["degrees_of_freedom"],
        "initial_temperature_kelvin":full[1]["temperature_kelvin"],
        "max_effective_energy_drift_ev":full[-1]["max_absolute_effective_energy_drift_ev"]}
    (a.work/"report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
