"""Essential final engine GPU-loss forwarding and v8 restart acceptance.
Exclusive GPU window required. Existing synthetic input; no benchmark or retries.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from validate_indexed_startup_integration import read, write, sha


def comparable(state):
    state = deepcopy(state)
    state["config"].pop("steps")
    state["config"].pop("fixed_geometry_cache", None)
    return state


def values(model):
    return [v for net in [n for row in model["descriptor"]["embedding"] for n in row] + model["fitting"]
            for layer in net["layers"] for row in [*layer["weights"], layer["bias"], layer["timestep"] or []]
            for v in row]


def difference(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        return max((difference(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        return max((difference(x, y) for x, y in zip(a, b)), default=0.)
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        assert math.isfinite(a) and math.isfinite(b)
        return abs(a - b)
    assert a == b
    return 0.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--baseline-work", type=Path, required=True, help="Accepted tiled-integrated-01 with historical v7 outputs")
    p.add_argument("--work", type=Path, required=True)
    a = p.parse_args()
    engine, baseline, work = [v.resolve() for v in (a.engine, a.baseline_work, a.work)]
    trainer = engine.with_name("omnimd-native-train" + engine.suffix)
    assert engine.is_file() and trainer.is_file()
    started = datetime.now(timezone.utc).isoformat()
    work.mkdir(parents=True, exist_ok=False)
    for name in ("initial.json", "dataset.jsonl"):
        shutil.copyfile(baseline / name, work / name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
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
            raise RuntimeError(f"{label} failed; stop GPU work without retry: {result.stderr}")
        return result.stdout

    def train(label, request, error=None):
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
        assert report["native_v1_reload_max_abs_energy_force_virial"] == [0, 0, 0]
        assert state["dataset_sha256"] == sha(work / "dataset.jsonl")
        profile = report["cuda_gradient_profile"]
        gpu = request["config"].get("cuda_loss", "cpu") == "gpu"
        assert profile["kernel_launches"] == (10 if gpu else 4) * profile["tiles"]
        assert profile["synchronizations"] == profile["tiles"]
        assert profile["max_frames_per_launch"] == 3 and profile["max_atoms_per_launch"] <= 12
        assert profile["peak_device_buffer_bytes"] <= 64 * 1024 * 1024
        assert profile["peak_host_tile_payload_bytes"] <= 32 * 1024 * 1024
        if gpu:
            assert state["schema"] == "omnimd-native-full-checkpoint-v8"
            assert state["execution"]["backend"] == "cuda_tiled_device_loss_f64_v1"
            assert profile["device_prediction_frames"] == profile["frames"]
            assert profile["prediction_and_seeds_ms"] == 0.
            assert report["cuda_tiling"]["device_loss"]["extra_device_bytes"] > 0
        else:
            assert state["schema"] == "omnimd-native-full-checkpoint-v7"
            assert state["execution"]["backend"] == "cuda_tiled_reverse_f64_v1"
            assert profile["device_prediction_frames"] == 0
        runs[label] = {"schema": state["schema"], "execution": state["execution"],
                       "profile": profile, "rates": report["learning_rate"],
                       "reload_max_abs_energy_force_virial": [0, 0, 0],
                       "model_sha256": sha(folder / "model.json"),
                       "checkpoint_sha256": sha(folder / "checkpoint.json")}
        return state, report

    caps = json.loads(execute("capabilities", [engine, "capabilities"]))
    assert "v8" in caps["training"]["cuda"]["loss_preparation"]["checkpoint"]
    assert caps["training"]["cuda"]["tiling"]["gpu_loss_kernels_per_tile"] == 10
    request = read(baseline / "tiled.request.json")
    request.update(initial_model="initial.json", dataset="dataset.jsonl")
    assert request["device"] == "cuda:0" and request["config"]["steps"] == 7
    assert "cuda_loss" not in request["config"]
    old, old_report = train("default-cpu-loss", request)
    historical = read(baseline / "tiled/checkpoint.json")
    assert old == historical
    assert (work / "default-cpu-loss/model.json").read_bytes() == (baseline / "tiled/model.json").read_bytes()
    metric_keys = ("initial_training", "final_training", "initial_validation", "final_validation")
    for key in metric_keys:
        assert old_report[key] == read(baseline / "tiled/report.json")[key]
    request["config"]["cuda_loss"] = "gpu"
    state, report = train("gpu-loss", request)
    assert state["execution"]["program_sha256"] != old["execution"]["program_sha256"]
    numeric_delta = difference(values(state["model"]) + state["first_moment"] + state["second_moment"],
                               values(old["model"]) + old["first_moment"] + old["second_moment"])
    assert numeric_delta < 3e-12
    metadata = []
    for source in (old, state):
        copy = deepcopy(source)
        for key in ("schema", "execution", "model", "first_moment", "second_moment"):
            copy.pop(key)
        copy["config"].pop("cuda_loss", None)
        metadata.append(copy)
    assert metadata[0] == metadata[1]
    metric_deltas = {key: difference(report[key], old_report[key]) for key in metric_keys}
    assert max(metric_deltas.values()) < 3e-12
    assert report["cuda_gradient_profile"]["frames"] == 43
    assert report["cuda_gradient_profile"]["tiles"] == 19
    assert report["minibatch"] == old_report["minibatch"]
    assert report["learning_rate"] == old_report["learning_rate"]
    resumed = deepcopy(request)
    resumed.pop("initial_model")
    resumed.update(checkpoint="gpu-loss/checkpoint-step-2.json", dataset_access="memory")
    resumed["config"].update(steps=5, fixed_geometry_cache=False)
    actual, actual_report = train("resume-v8-memory-cache-off", resumed)
    assert comparable(actual) == comparable(state)
    assert (work / "resume-v8-memory-cache-off/model.json").read_bytes() == (work / "gpu-loss/model.json").read_bytes()
    for key in ("final_training", "final_validation"):
        assert actual_report[key] == report[key]
    assert actual_report["learning_rate"]["first_update"] == 3
    assert actual_report["learning_rate"]["first_rate"] == 0.0025
    invalid = deepcopy(request)
    invalid["config"].pop("cuda_tiling")
    train("reject-missing-tiling", invalid, "requires explicit reverse cuda_tiling")
    invalid = deepcopy(resumed)
    invalid["config"].pop("cuda_loss")
    train("reject-loss-mode-change", invalid, "immutable configuration")
    invalid = deepcopy(resumed)
    invalid["checkpoint"] = str(baseline / "tiled/checkpoint-step-2.json")
    train("reject-v7-to-v8", invalid, "immutable configuration")
    receipt = {
        "schema": "omnimd-device-loss-integration-v1", "status": "passed",
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_native_commit": "36ee5cf4cc261437b6dcc03b182073dee72c5cae",
        "integrated_native_commit": "230279d2adb6151321720f27af20c45252f1f096",
        "source_base": "f6030da6cef085e37bd2d135878c591218f30b2e",
        "scope": "Essential engine GPU-loss forwarding, historical v7 equality, tolerant v8 comparison and exact v8 restart on existing 24-frame original synthetic input; no performance/capacity benchmark.",
        "frames": 24, "training_frames": 18, "validation_frames": 6, "optimizer_updates": 7,
        "default_v7_checkpoint_all_fields_model_bytes_metrics_and_ptx_exact": True,
        "gpu_loss_vs_cpu_loss_parameters_moments_max_abs": numeric_delta,
        "partition_metrics_max_abs_differences": metric_deltas,
        "comparison_tolerance": 3e-12,
        "v8_step_2_resume_memory_cache_off_exact_except_additional_steps_and_cache_flag": True,
        "v8_resume_model_bytes_and_final_metrics_exact": True,
        "tiling_plan": report["cuda_tiling"], "runs": runs,
        "rejected_before_output": ["missing tiling", "changed loss mode on resume", "historical v7 to v8"],
        "source_sha256": report["source_sha256"],
        "sha256": {name: sha(path) for name, path in [
            ("engine", engine), ("trainer", trainer), ("script", Path(__file__)),
            ("helper", Path(__file__).with_name("validate_indexed_startup_integration.py")),
            ("dataset", work / "dataset.jsonl"), ("initial_model", work / "initial.json"),
            ("historical_v7", baseline / "tiled/checkpoint.json")]},
        "checks": [{"label": c["label"], "exit_code": c["exit_code"]} for c in commands]}
    write(work / "acceptance.json", receipt)
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
