"""Essential engine large-frame forwarding and exact v9 continuation.
Uses B's existing original synthetic input. Exclusive GPU window required.
No capacity/derivative benchmark; stop on first unexpected failure without retry.
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
from validate_device_loss_integration import comparable, difference, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--baseline-work", required=True, type=Path,
                        help="Accepted original large-frame-replay-02 input directory")
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    engine, baseline, work = [p.resolve() for p in (args.engine, args.baseline_work, args.work)]
    trainer = engine.with_name("omnimd-native-train" + engine.suffix)
    assert engine.is_file() and trainer.is_file()
    started = datetime.now(timezone.utc).isoformat()
    work.mkdir(parents=True, exist_ok=False)
    for name in ("initial.json", "dataset.jsonl", "medium.json"):
        shutil.copyfile(baseline / name, work / name)
    assert sha(work / "dataset.jsonl") == "051ad1506b261e3450145a18a2f65951bc0b0e547319094b276e7d816368be4a"
    records = [json.loads(line) for line in (work / "dataset.jsonl").read_text().splitlines()]
    frames = [{"index": r["data"]["index"], "id": r["data"]["id"],
               "atoms": len(r["data"]["frame"]["positions"])}
              for r in records if r["kind"] == "frame"]
    assert len(frames) == 8 and {f["atoms"] for f in frames} == {32, 128, 256}
    assert records[-1]["frames"] == 8
    request = read(baseline / "large-gpu-loss.request.json")
    request.update(initial_model="initial.json", dataset="dataset.jsonl")
    request["config"].update(steps=4, checkpoint_every=2)
    assert request["config"]["cuda_tiling"] == {"max_frames": 4, "max_atoms": 512, "max_atoms_per_frame": 256}
    valid_ids = request["config"]["split"]["validation_ids"]
    assert len(valid_ids) == 2
    assert any(f["atoms"] == 256 for f in frames if f["id"] in valid_ids)
    assert any(f["atoms"] == 256 for f in frames if f["id"] not in valid_ids)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    commands, runs = [], {}

    def execute(label, command, error=None):
        result = subprocess.run([str(v) for v in command], capture_output=True,
                                text=True, timeout=120, env=env)
        for name in ("stdout", "stderr"):
            (work / (label + "." + name)).write_text(getattr(result, name), encoding="utf-8")
        commands.append({"label": label, "command": [str(v) for v in command], "exit_code": result.returncode})
        write(work / "commands.json", commands)
        if error:
            assert result.returncode and error in result.stderr, (label, result.stderr)
        elif result.returncode:
            raise RuntimeError(f"{label} failed; stop GPU work without retry: {result.stderr}")
        return result.stdout

    def train(label, q, error=None):
        q = deepcopy(q)
        q["output_directory"] = label
        path = work / (label + ".request.json")
        write(path, q)
        execute(label, [engine, "train-native", path], error)
        folder = work / label
        if error:
            assert not folder.exists()
            return None, None
        state, report = read(folder / "checkpoint.json"), read(folder / "report.json")
        gpu = q["config"]["cuda_loss"] == "gpu"
        profile = report["cuda_gradient_profile"]
        resources = report["cuda_tiling"]
        assert state["schema"] == "omnimd-native-full-checkpoint-v9"
        assert state["execution"]["backend"] == ("cuda_tiled_device_loss_f64_v2" if gpu else "cuda_tiled_reverse_f64_v1")
        assert state["dataset_sha256"] == sha(work / "dataset.jsonl")
        assert state["step"] == 4
        assert report["native_v1_reload_max_abs_energy_force_virial"] == [0., 0., 0.]
        assert profile["max_atoms_per_frame"] == 256
        assert 256 <= profile["max_atoms_per_launch"] <= resources["effective_max_atoms"] <= 512
        assert profile["max_frames_per_launch"] <= 4
        assert profile["kernel_launches"] == (10 if gpu else 4) * profile["tiles"]
        assert profile["synchronizations"] == profile["tiles"]
        assert profile["frames"] == (6 if "checkpoint" in q else 12)
        assert profile["device_prediction_frames"] == (profile["frames"] if gpu else 0)
        assert profile["peak_device_buffer_bytes"] <= 64 * 1024 * 1024
        assert profile["peak_host_tile_payload_bytes"] <= 32 * 1024 * 1024
        assert resources["estimated_scalar_terms_at_capacity"] <= 50_000_000
        if gpu:
            assert profile["prediction_and_seeds_ms"] == 0.
        runs[label] = {"schema": state["schema"], "execution": state["execution"],
                       "profile": profile, "resources": resources,
                       "minibatch": report["minibatch"], "learning_rate": report["learning_rate"],
                       "reload_max_abs_energy_force_virial": report["native_v1_reload_max_abs_energy_force_virial"],
                       "model_sha256": sha(folder / "model.json"), "checkpoint_sha256": sha(folder / "checkpoint.json")}
        return state, report

    caps = json.loads(execute("capabilities", [engine, "capabilities"]))["training"]["cuda"]
    assert caps["tiling"]["max_atoms_per_frame"] == 256
    assert caps["tiling"]["default_max_atoms_per_frame"] == 64
    assert "v9" in caps["large_frames"]["checkpoint"]
    bad = deepcopy(request)
    bad["config"]["cuda_tiling"].pop("max_atoms_per_frame")
    train("reject-default-frame-limit", bad, "configured atom limit 64")
    bad = deepcopy(request)
    bad["initial_model"] = "medium.json"
    train("reject-medium-work-budget", bad, "256 atoms; effective capacity 138")
    cpu_request = deepcopy(request)
    cpu_request["config"]["cuda_loss"] = "cpu"
    cpu_state, cpu_report = train("cpu-loss-v9", cpu_request)
    gpu_state, gpu_report = train("gpu-loss-v9", request)
    numeric_delta = difference(values(gpu_state["model"]) + gpu_state["first_moment"] + gpu_state["second_moment"],
                               values(cpu_state["model"]) + cpu_state["first_moment"] + cpu_state["second_moment"])
    assert numeric_delta < 1e-10
    metadata = []
    for source in (cpu_state, gpu_state):
        state = deepcopy(source)
        for key in ("execution", "model", "first_moment", "second_moment"):
            state.pop(key)
        state["config"].pop("cuda_loss", None)
        metadata.append(state)
    assert metadata[0] == metadata[1]
    metric_keys = ("initial_training", "final_training", "initial_validation", "final_validation")
    metric_deltas = {k: difference(cpu_report[k], gpu_report[k]) for k in metric_keys}
    assert max(metric_deltas.values()) < 1e-10
    assert cpu_report["minibatch"] == gpu_report["minibatch"]
    resume = deepcopy(request)
    resume.pop("initial_model")
    resume.update(checkpoint="gpu-loss-v9/checkpoint-step-2.json", dataset_access="memory")
    resume["config"].update(steps=2, fixed_geometry_cache=False)
    resumed_state, resumed_report = train("resume-v9-memory-cache-off", resume)
    assert comparable(resumed_state) == comparable(gpu_state)
    assert (work / "resume-v9-memory-cache-off/model.json").read_bytes() == (work / "gpu-loss-v9/model.json").read_bytes()
    for key in ("final_training", "final_validation"):
        assert resumed_report[key] == gpu_report[key]
    assert resumed_report["learning_rate"]["first_update"] == 3
    assert resumed_report["learning_rate"]["first_rate"] == .003
    assert resumed_report["learning_rate"]["last_rate"] == .003 * .8
    before = sha(work / "gpu-loss-v9/checkpoint-step-2.json")
    bad = deepcopy(resume)
    bad["config"]["cuda_tiling"]["max_atoms"] = 513
    train("reject-changed-tile-capacity", bad, "immutable configuration")
    assert sha(work / "gpu-loss-v9/checkpoint-step-2.json") == before
    receipt = {
        "schema": "omnimd-large-frame-integration-v1", "status": "passed",
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "native_commit": "c7ec6f187e296e0e4066ed4978d11a03a6dd08cd",
        "source_base": "cfe1680e850815d329110e3c1baa6fda16bcbb75",
        "scope": "Essential Windows engine forwarding and v9 resume using existing original synthetic input; no broad matrix, derivative or speed benchmark.",
        "frames": frames, "training_frames": 6, "validation_ids": valid_ids,
        "optimizer_updates": 4, "cross_mode_parameters_moments_max_abs": numeric_delta,
        "partition_metrics_max_abs_differences": metric_deltas, "comparison_tolerance": 1e-10,
        "v9_step_2_to_4_resume_exact_except_steps_and_cache_flag": True,
        "resume_model_bytes_and_final_metrics_exact": True,
        "rejected_before_output": ["default64", "466-parameter effective138 work capacity", "changed tile capacity on resume"],
        "runs": runs, "source_sha256": gpu_report["source_sha256"],
        "sha256": {name: sha(path) for name, path in [
            ("engine", engine), ("trainer", trainer), ("script", Path(__file__)),
            ("json_helper", Path(__file__).with_name("validate_indexed_startup_integration.py")),
            ("comparison_helper", Path(__file__).with_name("validate_device_loss_integration.py")),
            ("dataset", work / "dataset.jsonl"), ("initial_model", work / "initial.json"),
            ("medium_model", work / "medium.json")]},
        "checks": [{"label": c["label"], "exit_code": c["exit_code"]} for c in commands],
        "linux_scope": "Previous06439f1 Linux v7/v8 evidence unchanged; new v9 large-frame execution unverified."}
    write(work / "acceptance.json", receipt)
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
