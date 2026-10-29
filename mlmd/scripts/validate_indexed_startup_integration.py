"""CPU-only indexed startup acceptance against saved pre-optimization outputs.
Uses existing original synthetic inputs. No GPU, timing benchmark or retries.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--baseline-work", type=Path, required=True,
                        help="Saved indexed-integrated-01 CPU fresh and final-engine-cpu outputs")
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
    commands = []

    def train(label, request, error=None):
        assert request["device"] == "cpu"
        path = work / (label + ".request.json")
        write(path, request)
        command = [str(engine), "train-native", str(path)]
        result = subprocess.run(command, text=True, capture_output=True, timeout=120, env=env)
        for kind in ("stdout", "stderr"):
            (work / (label + "." + kind)).write_text(getattr(result, kind), encoding="utf-8")
        commands.append({"label": label, "command": command, "exit_code": result.returncode})
        write(work / "commands.json", commands)
        folder = work / request["output_directory"]
        if error:
            assert result.returncode and error in result.stderr, (label, result.stderr)
            assert not folder.exists(), label
            return None, None
        if result.returncode:
            raise RuntimeError(f"{label} failed; no retry: {result.stderr}")
        state, report = read(folder / "checkpoint.json"), read(folder / "report.json")
        assert state["schema"] == "omnimd-native-full-checkpoint-v5"
        assert report["native_v1_reload_max_abs_energy_force_virial"] == [0, 0, 0]
        timing, startup = report["timing"], report["startup_timing"]
        for value in startup.values():
            assert math.isfinite(value) and value >= 0
        assert 0 <= timing["initialization_ms"] <= timing["startup_ms"]
        assert math.isclose(startup["before_first_update_ms"],
                            startup["before_trainer_ms"] + timing["startup_ms"],
                            rel_tol=1e-12, abs_tol=1e-9)
        if request.get("dataset_access", "memory") == "indexed":
            access = report["dataset_access"]
            assert access["mode"] == "indexed_jsonl_v1" and access["frames"] == 24
            assert access["frames_decoded_per_read"] == 1
            assert timing["startup_frame_reads"] == (48 if "checkpoint" in request else 72)
        else:
            assert "dataset_access" not in report and "startup_frame_reads" not in timing
        return state, report

    metrics = ("initial_training", "final_training", "initial_validation", "final_validation")
    scenarios = {}

    def compare(label, old_label, state, report):
        old = read(baseline / old_label / "checkpoint.json")
        old_report = read(baseline / old_label / "report.json")
        assert state == old, label  # Every field, including requested steps and Adam/sampler.
        assert (work / label / "model.json").read_bytes() == (baseline / old_label / "model.json").read_bytes()
        for key in metrics + ("minibatch",):
            assert report[key] == old_report[key], (label, key)
        new_access, old_access = report["dataset_access"], old_report["dataset_access"]
        assert old_access["frame_reads"] - new_access["frame_reads"] == 48
        assert old_access["integrity_verifications"] == new_access["integrity_verifications"]
        scenarios[label] = {
            "all_checkpoint_fields_exact": True, "model_bytes_exact": True,
            "all_four_partition_metrics_and_minibatch_exact": True,
            "frame_reads_before": old_access["frame_reads"], "frame_reads_after": new_access["frame_reads"],
            "saved_frame_reads": 48, "startup_frame_reads": report["timing"]["startup_frame_reads"],
            "whole_file_checks_before": old_access["integrity_verifications"],
            "whole_file_checks_after": new_access["integrity_verifications"],
            "startup_timing": report["startup_timing"], "trainer_startup_ms": report["timing"]["startup_ms"],
            "reload_max_abs_energy_force_virial": report["native_v1_reload_max_abs_energy_force_virial"],
            "baseline_checkpoint_sha256": sha(baseline / old_label / "checkpoint.json"),
            "candidate_checkpoint_sha256": sha(work / label / "checkpoint.json")}

    fresh = read(baseline / "cpu-indexed.request.json")
    assert fresh["config"]["steps"] == 7 and fresh["config"]["minibatch"]["batch_size"] == 7
    fresh.update(initial_model="initial.json", dataset="dataset.jsonl", output_directory="fresh")
    state, report = train("fresh", fresh)
    compare("fresh", "cpu-indexed", state, report)
    resumed = read(baseline / "final-engine-cpu.request.json")
    assert resumed["config"]["steps"] == 1
    resumed.update(checkpoint=str(baseline / "cpu-indexed/checkpoint-step-6.json"),
                   dataset="dataset.jsonl", output_directory="resumed")
    resume_state, resume_report = train("resumed", resumed)
    compare("resumed", "final-engine-cpu", resume_state, resume_report)
    memory = deepcopy(fresh)
    memory.pop("dataset_access")
    memory["output_directory"] = "default-memory"
    memory_state, memory_report = train("default-memory", memory)
    assert memory_state == state
    assert (work / "default-memory/model.json").read_bytes() == (work / "fresh/model.json").read_bytes()
    for key in metrics + ("minibatch",):
        assert memory_report[key] == report[key], key
    zero = deepcopy(fresh)
    zero["config"]["steps"] = 0
    zero["output_directory"] = "rejected-zero-steps"
    train("rejected-zero-steps", zero, "steps must be 1..100000")
    cache = deepcopy(fresh)
    cache["config"]["fixed_geometry_cache"] = True
    cache["output_directory"] = "rejected-cpu-cache"
    train("rejected-cpu-cache", cache, "requires explicit CUDA full training")
    corrupt = read(baseline / "cpu-indexed/checkpoint-step-6.json")
    corrupt["first_moment"].pop()
    write(work / "malformed-checkpoint.json", corrupt)
    bad = deepcopy(resumed)
    bad.update(checkpoint="malformed-checkpoint.json", output_directory="rejected-moments")
    train("rejected-moments", bad, "checkpoint split or Adam state mismatch")
    receipt = {
        "schema": "omnimd-indexed-startup-integration-v1", "status": "passed",
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_native_commit": "cdf66b43e8899466d4c4cb4fb23a9c874af66a57",
        "integrated_native_commit": "958b362cd3fab0d22ac2c543ccd6503a8bbe30df",
        "source_base": "798d2d496dee1471539a43418c31abe66f3e4ab0",
        "scope": "CPU-only short 24-frame synthetic CLI integration; no GPU, capacity or timing benchmark. Single-run timing fields are diagnostic only.",
        "device": "cpu", "cuda_used": False, "frames": 24, "training_frames": 18, "validation_frames": 6,
        "checkpoint_schema": state["schema"], "scenarios": scenarios,
        "default_memory_all_checkpoint_fields_model_bytes_and_four_metrics_exact": True,
        "rejected_before_output": ["zero steps", "CPU geometry cache", "malformed checkpoint moments"],
        "source_sha256": report["source_sha256"],
        "sha256": {name: sha(path) for name, path in [
            ("engine", engine), ("trainer", trainer), ("script", Path(__file__)),
            ("dataset", work / "dataset.jsonl"), ("initial_model", work / "initial.json"),
            ("model", work / "fresh/model.json"),
            ("historical_step_6", baseline / "cpu-indexed/checkpoint-step-6.json")]},
        "checks": [{"label": c["label"], "exit_code": c["exit_code"]} for c in commands]}
    write(work / "acceptance.json", receipt)
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
