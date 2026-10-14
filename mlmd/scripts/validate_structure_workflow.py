#!/usr/bin/env python3
"""Small actual XYZ -> readout fit -> minimize -> MD workflow plus C API minimization.

Uses existing public/synthetic inputs. No downloads, new architectures or physical
accuracy claims. Output directory must be new. NumPy checks only comparisons.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def write_xyz(path, data):
    with path.open("x", encoding="utf8") as out:
        for entry in data:
            f, y = entry["frame"], entry["labels"]
            out.write(str(len(f["species"])) + "\n")
            lattice = " ".join(format(x, ".17e") for row in f["cell"] for x in row)
            virial = " ".join(format(y["virial"][i][j], ".17e") for j in range(3) for i in range(3))
            out.write(f'Lattice="{lattice}" Properties=species:S:1:pos:R:3:forces:R:3 pbc="T T T" '
                      f'energy={y["energy"]:.17e} virial="{virial}" units=EvAngstrom config_type=synthetic_teacher\n')
            for s, r, force in zip(f["species"], f["positions"], y["forces"]):
                out.write(s + " " + " ".join(format(x, ".17e") for x in r + force) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--public-request", type=Path, required=True)
    parser.add_argument("--public-native-model", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)

    def call(command, config, name, allow_nonconverged=False):
        request = args.work / (name + ".request.json")
        output = args.work / (name + ".jsonl")
        request.write_text(json.dumps(config), encoding="utf8")
        cmd = [str(args.binary), command, str(request)]
        if command != "train-native":
            cmd.append(str(output))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        (args.work / (name + ".log")).write_text(proc.stdout + proc.stderr, encoding="utf8")
        if allow_nonconverged:
            result = records(output)[-1]
            assert result["kind"] == "completed", proc.stderr
            assert (proc.returncode == 0) == result["converged"]
        else:
            assert proc.returncode == 0, proc.stderr
        return output

    original = records(args.training_dataset)
    selected = [r["data"] for r in original if r["kind"] == "frame" and r["data"]["frame"]["cell"] is not None]
    assert len(selected) >= 8
    # An explicit synthetic periodic subset, not an automatic training split.
    xyz_path = args.work / "teacher.extxyz"
    write_xyz(xyz_path, selected)
    dataset = call("import-xyz", {"input": str(xyz_path), "format": "Extxyz", "units": "EvAngstrom",
        "pbc": "Periodic", "type_map": original[0]["header"]["type_map"]}, "import")
    imported = records(dataset)
    for a, b in zip(selected, imported[1:-1]):
        assert a["frame"] == b["data"]["frame"]
        assert a["labels"] == b["data"]["labels"]
    initial_options = {"native_model": str(args.initial_model)}
    initial = records(call("batch-eval", {"options": initial_options, "dataset": str(dataset)}, "initial"))
    nvalidation = max(1, len(selected)//4)
    validation_ids = [f"xyz/{i}" for i in range(len(selected)-nvalidation, len(selected))]
    trained_dir = args.work / "trained"
    call("train-native", {"schema": "omnimd-native-readout-train-v1", "initial_model": str(args.initial_model),
        "dataset": str(dataset), "output_directory": str(trained_dir),
        "config": {"split": {"mode": "explicit", "validation_ids": validation_ids}, "fit_bias": False,
        "energy_weight": 1., "force_weight": 1., "energy_scale_ev_per_atom": 1.,
        "force_scale_ev_per_angstrom": .01, "ridge_inverse_ev2": 0., "rank_rtol": 1e-12}}, "train")
    trained_options = {"native_model": str(trained_dir / "model.json")}
    fitted = records(call("batch-eval", {"options": trained_options, "dataset": str(dataset)}, "fitted"))
    for metric in ["energy_per_atom", "force_components"]:
        assert fitted[-1]["metrics"][metric]["rmse"] < 1e-10

    settings = {"force_tolerance_ev_per_angstrom": 1e-4, "energy_tolerance_ev": 1e-7,
        "initial_step_angstrom2_per_ev": .1, "max_displacement_angstrom": .05,
        "max_backtracks": 32, "max_iterations": 300, "armijo": 1e-4}
    minimum = call("minimize", {"options": trained_options, "dataset_frame": {"dataset": str(dataset), "index": 0},
        "minimization": settings}, "trained-minimum", allow_nonconverged=True)
    minimum_data = records(minimum)
    assert minimum_data[-1]["final_energy_ev"] < minimum_data[-1]["initial_energy_ev"]
    for a, b in zip(minimum_data[1:-2], minimum_data[2:-1]):
        assert b["evaluation"]["energy"] <= a["evaluation"]["energy"]
    final_dataset = Path(minimum_data[-1]["final_dataset"])
    geometry = records(final_dataset)[1]["data"]
    assert geometry["frame"]["cell"] == selected[0]["frame"]["cell"]
    assert all(v is None for v in geometry["labels"].values())
    reimport = call("import-xyz", {"input": minimum_data[-1]["final_extxyz"], "format": "Extxyz",
        "units": "EvAngstrom", "pbc": "Periodic", "type_map": original[0]["header"]["type_map"]}, "reimport")
    assert records(reimport)[1]["data"]["frame"] == geometry["frame"]
    masses = [{"O": 15.999, "H": 1.008}[s] for s in geometry["frame"]["species"]]
    trajectory = records(call("run", {"options": trained_options, "dataset_frame": {"dataset": str(final_dataset), "index": 0},
        "masses": masses, "velocities": [[0.,0.,0.] for _ in masses], "dynamics_units": "EvAngstromFsU",
        "timestep_fs": .001, "steps": 5, "sample_every": 1}, "md"))
    assert trajectory[-1]["kind"] == "completed"

    public = read(args.public_request)
    public_settings = dict(settings, force_tolerance_ev_per_angstrom=.01, energy_tolerance_ev=1e-5,
                           initial_step_angstrom2_per_ev=.02, max_iterations=200)
    public_minima = {}
    for backend, options in [("capi", public["options"]), ("native", {"native_model": str(args.public_native_model)})]:
        path = call("minimize", {"options": options, "frame": public["frames"][2],
            "minimization": public_settings}, "public-" + backend, allow_nonconverged=True)
        values = records(path)
        assert values[-1]["final_energy_ev"] < values[-1]["initial_energy_ev"]
        public_minima[backend] = values
    capi_final, native_final = [public_minima[b][-2] for b in ["capi", "native"]]
    public_error = {key: float(np.max(np.abs(np.asarray(capi_final["evaluation"][key]) - np.asarray(native_final["evaluation"][key]))))
                    for key in ["energy","forces","virial"]}
    assert max(public_error.values()) < 1e-9, public_error
    limited = call("minimize", {"options": public["options"], "frame": public["frames"][2],
        "minimization": dict(public_settings,max_iterations=1)}, "limit", allow_nonconverged=True)
    assert records(limited)[-1]["status"] == "max_iterations"
    report = {"scope": "manufactured teacher workflow and public regression minimization; no physical accuracy claim",
        "training_frames_total": len(selected), "explicit_validation_ids": validation_ids,
        "initial_metrics": initial[-1]["metrics"], "fitted_metrics": fitted[-1]["metrics"],
        "trained_minimization": minimum_data[-1],
        "public_minimization": {b: data[-1] for b,data in public_minima.items()},
        "public_final_max_absolute_backend_difference": public_error,
        "md_steps": trajectory[-1]["steps"],
        "md_max_energy_drift_ev": trajectory[-1]["max_absolute_energy_drift_ev"],
        "checks": ["extxyz geometry/E/F/V roundtrip","explicit heldout IDs through unified training command",
            "both minimization backends decrease energy","nonconvergence status/exit","fixed cell",
            "geometry exports have no inferred reference labels","export reimport exact","minimized dataset enters MD"],
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest()}
    # Keep only portable result metadata in the committed report.
    for result in [report["trained_minimization"], *report["public_minimization"].values()]:
        result.pop("final_extxyz",None)
        result.pop("final_dataset",None)
    (args.work/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()