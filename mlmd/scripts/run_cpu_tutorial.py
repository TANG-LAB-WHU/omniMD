"""Run the documented CPU-only workflow with original synthetic data (stdlib only).

Build omnimd-mlmd and omnimd-native binaries first. This script never builds,
installs, downloads, loads a GPU, or overwrites an existing work directory.
It writes every request so individual CLI steps can be inspected and reused.
The generated teacher is a software fixture, not a physical force field.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--eval-exe", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    engine, evaluator, work = args.engine.resolve(), args.eval_exe.resolve(), args.work.resolve()
    trainer = engine.with_name("omnimd-native-train.exe" if engine.suffix == ".exe" else "omnimd-native-train")
    for exe in (engine, evaluator, trainer):
        if not exe.is_file():
            raise ValueError(f"Build required binary first: {exe}")
    work.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    commands = []

    def run(label, command, input_text=None):
        command = [str(item) for item in command]
        print(f"{len(commands)+1}. {label}", flush=True)
        completed = subprocess.run(command, input=input_text, text=True, encoding="utf-8",
                                   capture_output=True, timeout=120, check=True)
        commands.append({"step": label, "command": command, "exit_code": completed.returncode})
        (work / f"step-{len(commands):02d}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (work / f"step-{len(commands):02d}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        return completed.stdout

    def request(name, value):
        path = work / name
        write(path, value)
        return path

    teacher = read(source / "native/fixtures/synthetic-v1.json")
    initial = deepcopy(teacher)
    for kind, net in enumerate(teacher["fitting"]):
        for index, weight in enumerate(net["layers"][-1]["weights"]):
            weight[0] = 0.8 * math.sin((kind * 7 + index + 1) * 0.7)
    for net in initial["fitting"]:
        for weight in net["layers"][-1]["weights"]:
            weight[0] = 0.0
    write(work / "teacher.json", teacher)
    write(work / "initial.json", initial)
    generator = random.Random(20261016)
    base = read(source / "native/fixtures/frame.jsonl")
    frames = []
    for index in range(48):
        frame = deepcopy(base)
        frame["cell"] = None
        for position in frame["positions"]:
            for axis in range(3):
                position[axis] += generator.uniform(-0.24, 0.24)
        if index % 3 == 0:
            frame["positions"].pop()
            frame["species"].pop()
        frames.append(frame)
    lines = "".join(json.dumps(frame, allow_nan=False) + "\n" for frame in frames)
    labels = [json.loads(line) for line in run("Generate synthetic CPU teacher E/F/V",
              [evaluator, work / "teacher.json"], lines).splitlines()]
    if len(labels) != len(frames):
        raise AssertionError("Teacher did not return one evaluation per frame")
    with (work / "training.extxyz").open("x", encoding="utf-8", newline="\n") as handle:
        for frame, label in zip(frames, labels):
            handle.write(str(len(frame["species"])) + "\n")
            virial = " ".join(repr(label["virial"][row][col]) for col in range(3) for row in range(3))
            handle.write('Properties=species:S:1:pos:R:3:forces:R:3 pbc="F F F" '
                         f'energy={label["energy"]!r} virial="{virial}" units=EvAngstrom\n')
            for species, position, force in zip(frame["species"], frame["positions"], label["forces"]):
                handle.write(species + " " + " ".join(repr(x) for x in position + force) + "\n")
    options = {"native_model": str(work / "initial.json"), "device": "cpu"}
    write(work / "capabilities.json", json.loads(run("Read static capabilities", [engine, "capabilities"])))
    info = request("model-info.request.json", {"options": options})
    run("Load native CPU model once and report metadata", [engine, "model-info", info, work / "model-info.json"])
    imp = request("import.request.json", {"input": str(work / "training.extxyz"), "format": "Extxyz",
                  "type_map": teacher["type_map"], "units": "EvAngstrom", "pbc": "NonPeriodic"})
    run("Import explicit extxyz labels", [engine, "import-xyz", imp, work / "dataset.jsonl"])
    run("Inspect the complete dataset without loading a model",
        [engine, "inspect-dataset", work / "dataset.jsonl", work / "inspection.json"])
    inspection = read(work / "inspection.json")
    if not inspection["labels"]["all_frames_have_energy_and_forces"]:
        raise AssertionError("Tutorial training labels are missing")
    if inspection["dataset_sha256"] != digest(work / "dataset.jsonl"):
        raise AssertionError("Inspection byte hash mismatch")
    dataset = records(work / "dataset.jsonl")
    for record, label in zip(dataset[1:-1], labels):
        if record["data"]["labels"] != {key: label[key] for key in ("energy", "forces", "virial")}:
            raise AssertionError("Import changed synthetic labels or tensor order")
    train = request("train.request.json", {"schema": "omnimd-native-readout-train-v1",
        "initial_model": "initial.json", "dataset": "dataset.jsonl", "output_directory": "trained",
        "config": {"split": {"mode": "explicit", "validation_ids": [f"xyz/{i}" for i in range(0,48,4)]},
            "fit_bias": False, "energy_weight": 1.0, "force_weight": 1.0,
            "energy_scale_ev_per_atom": 1.0, "force_scale_ev_per_angstrom": 0.01,
            "ridge_inverse_ev2": 0.0, "rank_rtol": 1e-12}})
    run("Train final readout through the unified CPU CLI", [engine, "train-native", train])
    report = read(work / "trained/report.json")
    if report["final_validation"]["normalized_ef_objective"] >= report["initial_validation"]["normalized_ef_objective"] * 1e-8:
        raise AssertionError("Manufactured held-out error did not improve")
    options = {"native_model": str(work / "trained/model.json"), "device": "cpu"}
    batch = request("batch.request.json", {"options": options, "dataset": str(work / "dataset.jsonl")})
    run("Evaluate every frame with one retained model", [engine, "batch-eval", batch, work / "predictions.jsonl"])
    predictions = records(work / "predictions.jsonl")
    if predictions[-1].get("kind") != "completed" or predictions[-1]["frames"] != 48:
        raise AssertionError("Batch evaluation incomplete")
    md = request("md.request.json", {"options": options,
        "dataset_frame": {"dataset": str(work / "dataset.jsonl"), "index": 0},
        "masses": [15.999 if s == "O" else 1.008 for s in frames[0]["species"]],
        "dynamics_units": "EvAngstromFsU",
        "initialize_velocities": {"temperature_kelvin": 30, "seed": 42, "rescale_to_temperature": True},
        "remove_center_of_mass": True,
        "ensemble": {"kind": "nvt_csvr", "temperature_kelvin": 30, "tau_fs": 0.1, "seed": 43},
        "timestep_fs": 0.001, "steps": 10, "sample_every": 1, "checkpoint_every": 5})
    run("Validate MD and initial E/F/V", [engine, "check", md])
    run("Run ten CPU CSVR steps", [engine, "run", md, work / "md.jsonl"])
    resume = request("resume.request.json", {"options": options,
        "restart": str(work / "md.jsonl.step-5.checkpoint.json"),
        "timestep_fs": 0.001, "steps": 5, "sample_every": 1})
    run("Resume five steps with saved thermostat RNG", [engine, "run", resume, work / "resumed.jsonl"])
    full, resumed = records(work / "md.jsonl"), records(work / "resumed.jsonl")
    if full[-1].get("kind") != "completed" or resumed[-1].get("kind") != "completed":
        raise AssertionError("MD did not complete")
    if [r for r in full if r.get("kind") == "frame" and r["step"] >= 5] != [r for r in resumed if r.get("kind") == "frame"]:
        raise AssertionError("Restarted snapshots differ")
    if read(work / "md.jsonl.checkpoint.json") != read(work / "resumed.jsonl.checkpoint.json"):
        raise AssertionError("Final checkpoints differ")
    receipt = {"schema": "omnimd-cpu-tutorial-receipt-v1", "status": "passed",
        "scope": "48 synthetic nonperiodic frames; software workflow only; CPU readout training and short NVT/restart; no physical accuracy claim",
        "frames": 48, "training_frames": len(report["training_frames"]),
        "validation_frames": len(report["validation_frames"]), "data_rank": report["data_rank"],
        "initial_validation": report["initial_validation"], "final_validation": report["final_validation"],
        "inspection_sha_verified": True, "extxyz_labels_preserved": True,
        "restart_snapshots_and_checkpoint_exact": True, "commands": commands,
        "sha256": {name: digest(work / name) for name in ("dataset.jsonl", "teacher.json", "initial.json", "trained/model.json", "training.extxyz")},
        "executable_sha256": {"engine": digest(engine), "evaluator": digest(evaluator), "trainer": digest(trainer)},
        "script_sha256": digest(__file__)}
    write(work / "receipt.json", receipt)
    print(f"Passed. Requests, reports, checkpoints and receipt: {work}")


if __name__ == "__main__":
    main()
