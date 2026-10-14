#!/usr/bin/env python3
"""Public/synthetic dataset workflow and bounded two-backend MD comparison.

Requires NumPy only for generating NPY and independent metric arithmetic.
It does not train, download, modify an input model, or run production data.
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
    values = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    assert values[-1]["kind"] == "completed", path
    return values


def max_error(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--public-request", type=Path, required=True)
    parser.add_argument("--public-evaluation", type=Path, required=True)
    parser.add_argument("--native-model", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    original = read(args.public_request)
    frames = original["frames"]
    reference = read(args.public_evaluation)["evaluations"]
    capi_options = original["options"]
    native_options = {"native_model": str(args.native_model.resolve())}

    def command(command, config, name, success=True):
        request = args.work / (name + ".request.json")
        output = args.work / (name + ".jsonl")
        request.write_text(json.dumps(config), encoding="utf8")
        proc = subprocess.run([str(args.binary), command, str(request), str(output)],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (args.work / (name + ".log")).write_text(proc.stdout, encoding="utf8")
        assert (proc.returncode == 0) == success, proc.stdout
        return output

    root = args.work / "periodic"
    root.mkdir()
    (root / "type.raw").write_text("0 1 1 0 1 1\n", encoding="utf8")
    (root / "type_map.raw").write_text("O H\n", encoding="utf8")
    labeled = root / "set.000"
    labeled.mkdir()
    np.save(labeled / "coord.npy", np.asarray([frames[i]["positions"] for i in [1, 2]], dtype="<f8").reshape(2, -1))
    np.save(labeled / "box.npy", np.asarray([frames[i]["cell"] for i in [1, 2]], dtype="<f8").reshape(2, 9))
    # Deliberate synthetic label offsets make the expected metrics nonzero.
    np.save(labeled / "energy.npy", np.asarray([reference[1]["energy"] + .2, reference[2]["energy"] - .4]))
    np.save(labeled / "force.npy", (np.asarray([reference[i]["forces"] for i in [1, 2]]) + .01).reshape(2, -1))
    np.save(labeled / "virial.npy", (np.asarray([reference[i]["virial"] for i in [1, 2]]) - .03).reshape(2, 9))
    unlabeled = root / "set.001"
    unlabeled.mkdir()
    np.save(unlabeled / "coord.npy", np.asarray(frames[1]["positions"], dtype="<f8").reshape(1, -1))
    np.save(unlabeled / "box.npy", np.asarray(frames[1]["cell"], dtype="<f8").reshape(1, 9))
    dataset = command("import-deepmd", {"system": str(root), "sets": ["set.000", "set.001"],
        "units": "EvAngstrom", "pbc": "Periodic"}, "dataset")
    imported = records(dataset)
    assert imported[-1]["frames"] == 3
    assert imported[1]["data"]["frame"] == frames[1]
    assert imported[2]["data"]["frame"] == frames[2]
    assert imported[3]["data"]["labels"] == {"energy": None, "forces": None, "virial": None}
    for name, value in imported[-1]["source_sha256"].items():
        assert digest(root / name) == value

    predictions = {}
    for backend, options in [("capi", capi_options), ("native", native_options)]:
        path = command("batch-eval", {"options": options, "dataset": str(dataset)}, backend + "-eval")
        predictions[backend] = records(path)
        command("model-info", {"options": options}, backend + "-info")
    caps = subprocess.run([str(args.binary), "capabilities"], check=True, capture_output=True, text=True)
    assert json.loads(caps.stdout)["datasets"]["training"] is False
    differences = {}
    for key in ["energy", "forces", "virial"]:
        differences[key] = max(max_error(a["evaluation"][key], b["evaluation"][key])
            for a, b in zip(predictions["capi"][1:-1], predictions["native"][1:-1]))
    assert max(differences.values()) < 1e-10, differences
    measured = predictions["capi"][-1]["metrics"]
    independent = {}
    for metric, key, factor in [("energy_total", "energy", 1), ("energy_per_atom", "energy", 6),
                                ("force_components", "forces", 1), ("virial_components", "virial", 1)]:
        residual = np.concatenate([
            ((np.asarray(p["evaluation"][key]) - np.asarray(p["labels"][key])) / factor).reshape(-1)
            for p in predictions["capi"][1:3]
        ])
        expected = {"count": residual.size, "mae": float(np.mean(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(residual**2))), "max_absolute": float(np.max(np.abs(residual)))}
        for key2, value in expected.items():
            assert abs(measured[metric][key2] - value) < 1e-13
        independent[metric] = expected
    assert measured["labeled_frames"] == {"energy": 2, "forces": 2, "virial": 2}
    assert predictions["capi"][0]["dataset_sha256"] == digest(dataset)

    nonperiodic = args.work / "nonperiodic"
    (nonperiodic / "set.000").mkdir(parents=True)
    (nonperiodic / "type.raw").write_text("0 1 1 0 1 1\n", encoding="utf8")
    (nonperiodic / "type_map.raw").write_text("O H\n", encoding="utf8")
    (nonperiodic / "nopbc").write_text("", encoding="utf8")
    np.save(nonperiodic / "set.000/coord.npy", np.asarray(frames[0]["positions"]).reshape(1, -1))
    no_pbc_dataset = command("import-deepmd", {"system": str(nonperiodic), "sets": ["set.000"],
        "units": "EvAngstrom", "pbc": "NonPeriodic"}, "nonperiodic")
    assert records(no_pbc_dataset)[1]["data"]["frame"]["cell"] is None
    unlabelled_result = records(command("batch-eval", {"options": native_options,
        "dataset": str(no_pbc_dataset)}, "unlabeled-eval"))
    assert unlabelled_result[-1]["metrics"]["energy_total"]["rmse"] is None

    truncated = args.work / "truncated.jsonl"
    truncated.write_text("\n".join(dataset.read_text().splitlines()[:-1]) + "\n", encoding="utf8")
    partial = command("batch-eval", {"options": native_options, "dataset": str(truncated)},
                      "reject-truncated", success=False)
    assert not any(json.loads(line)["kind"] == "completed" for line in partial.read_text().splitlines())
    command("import-deepmd", {"system": str(root), "sets": ["set.000"],
        "units": "HartreeBohr", "pbc": "Periodic"}, "reject-units", success=False)

    trajectories = {}
    request = {"dataset_frame": {"dataset": str(dataset), "index": 1},
        "masses": [15.999, 1.008, 1.008, 15.999, 1.008, 1.008],
        "velocities": [[.0001,0,0],[-.0002,.0001,0],[.0001,-.0001,0],
                       [-.0001,0,0],[.0002,-.0001,0],[-.0001,.0001,0]],
        "dynamics_units": "EvAngstromFsU", "timestep_fs": .01, "steps": 20, "sample_every": 1}
    for backend, options in [("capi", capi_options), ("native", native_options)]:
        trajectories[backend] = records(command("run", dict(request, options=options), backend + "-md"))
    trajectory_errors = {}
    a_frames = [r for r in trajectories["capi"] if r["kind"] == "frame"]
    b_frames = [r for r in trajectories["native"] if r["kind"] == "frame"]
    for field in ["energy", "forces", "virial"]:
        trajectory_errors[field] = max(max_error(a["evaluation"][field], b["evaluation"][field])
            for a, b in zip(a_frames, b_frames))
    trajectory_errors["positions"] = max(max_error(a["frame"]["positions"], b["frame"]["positions"])
        for a, b in zip(a_frames, b_frames))
    trajectory_errors["velocities"] = max(max_error(a["velocities"], b["velocities"])
        for a, b in zip(a_frames, b_frames))
    assert max(trajectory_errors.values()) < 1e-9, trajectory_errors
    drift = {backend: data[-1]["max_absolute_energy_drift_ev"] for backend, data in trajectories.items()}
    assert max(drift.values()) < 1e-6, drift
    report = {"scope": "public model numerical compatibility plus synthetic-offset labels; not chemical accuracy",
        "binary_sha256": digest(args.binary), "native_model_sha256": digest(args.native_model),
        "capi_model_sha256": digest(Path(capi_options["model"])),
        "dataset_frames": 3, "labeled_frames": 2, "batch_max_absolute_backend_difference": differences,
        "independent_metrics": independent,
        "short_nve": {"steps": 20, "timestep_fs": .01, "elapsed_fs": .2,
            "max_absolute_backend_difference": trajectory_errors, "max_absolute_energy_drift_ev": drift},
        "checks": ["real NumPy NPY import", "source SHA256", "both model-info backends", "static capabilities",
                   "independent metric arithmetic", "unlabeled nonperiodic", "truncation fails without completed",
                   "wrong units rejected", "dataset_frame MD entry"]}
    (args.work / "report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()