"""Opt-in CUDA frozen-readout CLI parity replay; original synthetic data only.

Requires an authorized GPU window, existing CUDA-enabled native executables and
NVIDIA runtime. No packages/downloads; generated files go to a new directory.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, text=True,
                          encoding="utf-8", stdout=subprocess.PIPE, **kwargs).stdout


def error(a, b):
    result = [0.0, 0.0, 0.0]
    if len(a) != len(b):
        raise AssertionError("evaluation count mismatch")
    for x, y in zip(a, b):
        result[0] = max(result[0], abs(x["energy"] - y["energy"]))
        for k, field in enumerate(("forces", "virial"), 1):
            if len(x[field]) != len(y[field]):
                raise AssertionError("evaluation shape mismatch")
            result[k] = max(result[k], max(abs(i-j) for r, s in zip(x[field], y[field]) for i, j in zip(r, s)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-exe", type=Path, required=True)
    parser.add_argument("--train-exe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="Explicit visible CUDA device")
    args = parser.parse_args()
    if not args.device.startswith("cuda:") or not args.device[5:].isascii() or not args.device[5:].isdecimal():
        raise ValueError("device must be cuda:N")
    evaluator, trainer = args.eval_exe.resolve(), args.train_exe.resolve()
    if not all(x.is_file() for x in (evaluator, trainer)):
        raise ValueError("native executables must exist")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    initial = json.loads((root / "fixtures/full-tanh-v1.json").read_text())
    teacher = deepcopy(initial)
    parameter = 0
    for net in teacher["fitting"]:
        for row in net["layers"][-1]["weights"]:
            parameter += 1
            row[0] += 0.3 * math.sin(parameter * 0.83)
        parameter += 1
        net["layers"][-1]["bias"][0] += 0.3 * math.sin(parameter * 0.83)
    write(output / "initial.json", initial)
    write(output / "teacher.json", teacher)
    base = json.loads((root / "fixtures/frame.jsonl").read_text())
    frames = []
    for index in range(32):
        frame = deepcopy(base)
        if index % 2 == 0:
            frame["cell"] = None
        if index % 3 == 0:
            frame["positions"].pop()
            frame["species"].pop()
        for atom, position in enumerate(frame["positions"]):
            for axis in range(3):
                position[axis] += 0.19 * math.sin((index + 1) * (atom * 3 + axis + 1) * 0.618)
        frames.append(frame)
    lines = "".join(json.dumps(f, allow_nan=False) + "\n" for f in frames)
    (output / "frames.jsonl").write_text(lines, encoding="utf-8")
    labels = [json.loads(x) for x in run([evaluator, output / "teacher.json"], input=lines).splitlines()]
    records = [{"kind":"header", "header":{"schema":"omnimd-dataset-jsonl-v1", "type_map":initial["type_map"],
        "units":"EvAngstrom", "virial_convention":"MinusDEpsilonRow", "source":"original synthetic readout teacher; mixed composition and PBC; no physical accuracy claim", "selected_sets":["synthetic"]}}]
    for i, (frame, label) in enumerate(zip(frames, labels)):
        records.append({"kind":"frame", "data":{"index":i, "id":f"synthetic/{i}", "frame":frame,
            "labels":{k:label[k] for k in ("energy", "forces", "virial")}}})
    records.append({"kind":"completed", "frames":len(frames), "source_sha256":{
        "teacher.json":sha(output / "teacher.json"), "replay_cuda_readout.py":sha(Path(__file__))}})
    dataset = output / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records), encoding="utf-8")
    request = {"schema":"omnimd-native-readout-train-v1", "initial_model":"initial.json", "dataset":"dataset.jsonl",
        "config":{"split":{"mode":"explicit", "validation_ids":[f"synthetic/{i}" for i in range(24,32)]},
        "fit_bias":True, "energy_weight":1.0, "force_weight":1.0, "energy_scale_ev_per_atom":0.1,
        "force_scale_ev_per_angstrom":0.01, "ridge_inverse_ev2":0.0, "rank_rtol":1e-12}}
    reports, predictions = {}, {}
    for name, device in (("cpu", "cpu"), ("cuda", args.device), ("cuda-repeat", args.device)):
        request.update(device=device, output_directory=name)
        path = output / f"{name}.request.json"
        write(path, request)
        run([trainer, path])
        reports[name] = json.loads((output / name / "report.json").read_text())
        predictions[name] = [json.loads(x) for x in run([evaluator, output / name / "model.json"], input=lines).splitlines()]
        fitted = json.loads((output / name / "model.json").read_text())
        frozen = deepcopy(fitted)
        for net, old in zip(frozen["fitting"], initial["fitting"]):
            net["layers"][-1]["weights"] = old["layers"][-1]["weights"]
            net["layers"][-1]["bias"] = old["layers"][-1]["bias"]
        if frozen != initial:
            raise AssertionError("frozen parameters changed")
        if reports[name]["dataset_sha256"] != sha(dataset):
            raise AssertionError("actual dataset hash mismatch")
        if reports[name]["final_validation"]["normalized_ef_objective"] >= 1e-20:
            raise AssertionError("held-out synthetic recovery failed")
    repeat_exact = (output / "cuda/model.json").read_bytes() == (output / "cuda-repeat/model.json").read_bytes()
    difference = error(predictions["cpu"], predictions["cuda"])
    teacher_difference = error(labels, predictions["cuda"])
    if not repeat_exact or max(difference + teacher_difference) > 1e-9:
        raise AssertionError("CUDA fitted model or repeat mismatch")
    receipt = {"scope":"GPU-assisted frozen final affine weights and biases; E/F loss, CPU QR and assessment; not full-network GPU training",
        "frames":32, "train_frames":24, "validation_frames":8, "fit_bias":True,
        "same_device_repeat_model_exact":repeat_exact, "frozen_parameters_unchanged":True,
        "max_abs_cpu_cuda_fitted_efv":difference, "max_abs_cuda_teacher_efv":teacher_difference,
        "reports":{key:{k:report[k] for k in ("execution", "data_rank", "solve_rank", "initial_training", "final_training",
            "initial_validation", "final_validation", "native_v1_reload_max_abs_energy_force_virial")} for key, report in reports.items()},
        "sha256":{"dataset":sha(dataset), "initial_model":sha(output / "initial.json"),
            "cuda_model":sha(output / "cuda/model.json"), "trainer":sha(trainer), "evaluator":sha(evaluator), "replay":sha(Path(__file__))}}
    write(output / "acceptance.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
