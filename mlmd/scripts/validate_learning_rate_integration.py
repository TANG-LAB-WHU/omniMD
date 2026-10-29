"""CPU-only final engine forwarding and learning-rate boundary acceptance.
Reuses the accepted 24-frame synthetic indexed-startup inputs, with no GPU/benchmark.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from validate_indexed_startup_integration import read, write, sha


def comparable(state):
    state = deepcopy(state)
    state["config"].pop("steps")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--baseline-work", type=Path, required=True,
                        help="Accepted indexed-startup-integrated-01 with fresh CPU output")
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    engine, baseline, work = [v.resolve() for v in (args.engine, args.baseline_work, args.work)]
    trainer = engine.with_name("omnimd-native-train" + engine.suffix)
    assert engine.is_file() and trainer.is_file()
    started = datetime.now(timezone.utc).isoformat()
    work.mkdir(parents=True, exist_ok=False)
    for name in ("initial.json", "dataset.jsonl"):
        shutil.copyfile(baseline / name, work / name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    commands, runs = [], {}

    def execute(label, command, error=None):
        result = subprocess.run([str(v) for v in command], capture_output=True,
                                text=True, timeout=120, env=env)
        for kind in ("stdout", "stderr"):
            (work / (label + "." + kind)).write_text(getattr(result, kind), encoding="utf-8")
        commands.append({"label": label, "command": [str(v) for v in command], "exit_code": result.returncode})
        write(work / "commands.json", commands)
        if error:
            assert result.returncode and error in result.stderr, (label, result.stderr)
        elif result.returncode:
            raise RuntimeError(f"{label} failed; no retry: {result.stderr}")
        return result.stdout

    def train(label, request, error=None):
        assert request["device"] == "cpu"
        request = deepcopy(request)
        request["output_directory"] = label
        path = work / (label + ".request.json")
        write(path, request)
        execute(label, [engine, "train-native", path], error)
        folder = work / label
        if error:
            assert not folder.exists()
            return None, None
        state, report = read(folder / "checkpoint.json"), read(folder / "report.json")
        schedule = request["config"].get("learning_rate_schedule") or {"kind": "constant"}
        start, end = report["step_start"], report["step_end"]
        base = request["config"]["learning_rate"]
        def rate(k):
            return base * schedule["decay_rate"] ** (k // schedule["decay_steps"]) if schedule["kind"] == "exponential" else base
        assert report["learning_rate"] == {
            "contract": "omnimd-optimizer-update-learning-rate-v1", "schedule": schedule,
            "base_rate": base, "first_update": start + 1, "last_update": end,
            "first_rate": rate(start), "last_rate": rate(end - 1)}
        assert report["native_v1_reload_max_abs_energy_force_virial"] == [0, 0, 0]
        assert state["dataset_sha256"] == sha(work / "dataset.jsonl")
        runs[label] = {"rates": report["learning_rate"], "step_start": start, "step_end": end,
                       "schema": state["schema"], "reload_max_abs_energy_force_virial": [0, 0, 0],
                       "checkpoint_sha256": sha(folder / "checkpoint.json"), "model_sha256": sha(folder / "model.json")}
        return state, report

    capabilities = json.loads(execute("capabilities", [engine, "capabilities"]))
    assert capabilities["training"]["learning_rate"]["kinds"] == ["constant", "exponential"]
    assert "v6" in capabilities["training"]["learning_rate"]["checkpoint"]
    request = read(baseline / "fresh.request.json")
    assert request["device"] == "cpu" and request["config"]["steps"] == 7
    assert request["config"]["learning_rate"] == 0.005
    request.update(initial_model="initial.json", dataset="dataset.jsonl")
    schedule = {"kind": "exponential", "decay_steps": 2, "decay_rate": 0.5}
    request["config"]["learning_rate_schedule"] = schedule
    state, report = train("scheduled", request)
    assert state["schema"] == "omnimd-native-full-checkpoint-v6"
    old = read(baseline / "fresh/checkpoint.json")
    assert old["schema"] == "omnimd-native-full-checkpoint-v5"
    assert state["model"] != old["model"] and state["sampler"] == old["sampler"]
    for split in (2, 3):
        resumed = deepcopy(request)
        resumed.pop("initial_model")
        resumed["checkpoint"] = f"scheduled/checkpoint-step-{split}.json"
        resumed["config"]["steps"] = 7 - split
        actual, actual_report = train(f"resume-{split}", resumed)
        assert comparable(actual) == comparable(state)
        assert (work / f"resume-{split}/model.json").read_bytes() == (work / "scheduled/model.json").read_bytes()
        for key in ("final_training", "final_validation"):
            assert actual_report[key] == report[key]
    constant = deepcopy(request)
    constant["config"]["learning_rate_schedule"] = {"kind": "constant"}
    constant_state, constant_report = train("explicit-constant", constant)
    assert constant_state["schema"] == "omnimd-native-full-checkpoint-v6"
    normalized = deepcopy(constant_state)
    normalized["schema"] = old["schema"]
    normalized["config"].pop("learning_rate_schedule")
    assert normalized == old
    assert (work / "explicit-constant/model.json").read_bytes() == (baseline / "fresh/model.json").read_bytes()
    for key in ("initial_training", "final_training", "initial_validation", "final_validation"):
        assert constant_report[key] == read(baseline / "fresh/report.json")[key]
    legacy = deepcopy(request)
    legacy.pop("initial_model")
    legacy["checkpoint"] = str(baseline / "fresh/checkpoint-step-6.json")
    legacy["config"].update(steps=1, learning_rate_schedule=None)
    legacy_state, legacy_report = train("legacy-null-resume", legacy)
    assert comparable(legacy_state) == comparable(old)
    assert "learning_rate_schedule" not in legacy_state["config"]
    for key in ("final_training", "final_validation"):
        assert legacy_report[key] == constant_report[key]
    invalid = deepcopy(request)
    invalid["config"]["learning_rate_schedule"]["decay_rate"] = 1.1
    train("reject-invalid-rate", invalid, "finite decay_rate in (0,1]")
    for name in ("changed", "removed"):
        invalid = deepcopy(request)
        invalid.pop("initial_model")
        invalid["checkpoint"] = "scheduled/checkpoint-step-2.json"
        invalid["config"]["steps"] = 5
        if name == "changed":
            invalid["config"]["learning_rate_schedule"]["decay_rate"] = 0.25
        else:
            invalid["config"].pop("learning_rate_schedule")
        train("reject-" + name + "-schedule", invalid, "immutable configuration")
    receipt = {
        "schema": "omnimd-learning-rate-integration-v1", "status": "passed",
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_native_commit": "63ea457d7084d17dbd2db25e4c7ee42ca46740d5",
        "integrated_native_commit": "7fdb1e24709d79e949be9bc30ae8b410d9cf65ac",
        "source_base": "e0decf3f8e64c8bad10f79323fa036511f2088fa",
        "scope": "Final engine forwarding and CPU boundary/legacy acceptance on existing 24-frame synthetic input; no GPU or performance test; B broad permutations not repeated.",
        "cuda_used": False, "device": "cpu", "frames": 24, "training_frames": 18, "validation_frames": 6,
        "schedule": schedule, "expected_rates_updates_1_to_7": [0.005,0.005,0.0025,0.0025,0.00125,0.00125,0.000625],
        "v6_resume_splits_exact_except_additional_steps": [2,3],
        "v6_resume_model_bytes_and_final_metrics_exact": True,
        "constant_matches_historical_v5_except_explicit_schedule_and_schema": True,
        "null_schedule_resumes_historical_v5_exactly": True,
        "rejected_before_output": ["rate greater than one", "changed schedule on resume", "removed schedule on resume"],
        "runs": runs, "source_sha256": report["source_sha256"],
        "sha256": {name: sha(path) for name,path in [
            ("engine",engine),("trainer",trainer),("script",Path(__file__)),
            ("helper",Path(__file__).with_name("validate_indexed_startup_integration.py")),
            ("dataset",work/"dataset.jsonl"),("initial_model",work/"initial.json"),
            ("historical_v5",baseline/"fresh/checkpoint.json")]},
        "checks": [{"label": c["label"],"exit_code": c["exit_code"]} for c in commands]}
    write(work / "acceptance.json", receipt)
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
