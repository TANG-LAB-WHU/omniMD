"""Short final engine CUDA tiling/restart acceptance; exclusive GPU window required.
Uses saved original synthetic inputs. Stops on first failure; no retries/benchmark.
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


def comparable(state, cross_executor=False):
    state = deepcopy(state)
    state["config"].pop("steps")
    state["config"].pop("fixed_geometry_cache", None)
    if cross_executor:
        state.pop("schema")
        state.pop("execution")
        state["config"].pop("cuda_tiling", None)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--baseline-work", type=Path, required=True,
                   help="Accepted learning-rate-integrated-01 original CPU fixture")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--reuse-completed", action="store_true",
                   help="Reuse successful saved commands after a harness-only correction; never retry a failed command")
    a = p.parse_args()
    engine, baseline, work = [v.resolve() for v in (a.engine, a.baseline_work, a.work)]
    trainer = engine.with_name("omnimd-native-train" + engine.suffix)
    assert engine.is_file() and trainer.is_file()
    started = datetime.now(timezone.utc).isoformat()
    if a.reuse_completed:
        assert work.is_dir()
        saved_commands = read(work / "commands.json")
        assert all(c["exit_code"] == 0 for c in saved_commands), "No retry after a failed command"
    else:
        work.mkdir(parents=True, exist_ok=False)
        saved_commands = []
    for name in ("initial.json", "dataset.jsonl"):
        if a.reuse_completed:
            assert (baseline / name).read_bytes() == (work / name).read_bytes()
        else:
            shutil.copyfile(baseline / name, work / name)
    counts = [len(row["data"]["frame"]["positions"]) for row in
              (json.loads(line) for line in (work / "dataset.jsonl").read_text().splitlines())
              if row["kind"] == "frame"]
    assert len(counts) == 24
    atom_min, atom_max = min(counts), max(counts)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    commands, runs, reused = [], {}, []

    def execute(label, command, error=None):
        saved = [c for c in saved_commands if c["label"] == label]
        if saved:
            assert len(saved) == 1 and saved[0]["command"] == [str(v) for v in command]
            assert error is None and saved[0]["exit_code"] == 0
            commands.append(saved[0])
            reused.append(label)
            return (work / (label + ".stdout")).read_text(encoding="utf-8")
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
        if any(c["label"] == label for c in saved_commands):
            assert read(path) == request, "Saved request changed; do not reuse"
        else:
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
        assert profile["kernel_launches"] == 4 * profile["tiles"]
        assert profile["synchronizations"] == profile["tiles"]
        if request["config"].get("cuda_tiling"):
            assert state["schema"] == "omnimd-native-full-checkpoint-v7"
            assert state["execution"]["backend"] == "cuda_tiled_reverse_f64_v1"
            assert report["cuda_tiling"]["requested"] == {"max_frames": 3, "max_atoms": 16}
            assert 12 <= report["cuda_tiling"]["effective_max_atoms"] <= 16
            assert profile["max_frames_per_launch"] == 3
            assert 3 * atom_min <= profile["max_atoms_per_launch"] <= min(16, 3 * atom_max)
            assert profile["peak_device_buffer_bytes"] <= 64 * 1024 * 1024
            assert profile["peak_host_tile_payload_bytes"] <= 32 * 1024 * 1024
        else:
            assert state["schema"] == "omnimd-native-full-checkpoint-v6"
            assert state["execution"]["backend"] == "cuda_reverse_f64_v1"
            assert profile["tiles"] == profile["frames"] and profile["max_frames_per_launch"] == 1
        runs[label] = {"schema": state["schema"], "execution": state["execution"],
                       "profile": profile, "rates": report["learning_rate"],
                       "reload_max_abs_energy_force_virial": [0, 0, 0],
                       "model_sha256": sha(folder / "model.json"),
                       "checkpoint_sha256": sha(folder / "checkpoint.json")}
        return state, report

    caps = json.loads(execute("capabilities", [engine, "capabilities"]))
    tc = caps["training"]["cuda"]["tiling"]
    assert tc["max_frames"] == 32 and tc["max_atoms_per_tile"] == 1024
    assert tc["max_atoms_per_frame"] == 64 and tc["gradient_kernels_per_tile"] == 4
    request = read(baseline / "scheduled.request.json")
    request.update(device="cuda:0", initial_model="initial.json", dataset="dataset.jsonl")
    request["config"].update(cuda_gradient="reverse", fixed_geometry_cache=True)
    assert request["config"]["steps"] == 7 and request["config"]["minibatch"]["batch_size"] == 7
    old, old_report = train("framewise", request)
    request["config"]["cuda_tiling"] = {"max_frames": 3, "max_atoms": 16}
    state, report = train("tiled", request)
    assert comparable(state, True) == comparable(old, True)
    assert (work / "tiled/model.json").read_bytes() == (work / "framewise/model.json").read_bytes()
    for key in ("initial_training", "final_training", "initial_validation", "final_validation", "minibatch", "learning_rate"):
        assert report[key] == old_report[key], key
    assert report["cuda_gradient_profile"]["frames"] == 43
    assert report["cuda_gradient_profile"]["tiles"] == 19
    assert report["cuda_gradient_profile"]["kernel_launches"] == 76
    assert report["minibatch"]["partial_batches"] == 2
    resumed = deepcopy(request)
    resumed.pop("initial_model")
    resumed.update(checkpoint="tiled/checkpoint-step-2.json", dataset_access="memory")
    resumed["config"].update(steps=5, fixed_geometry_cache=False)
    actual, actual_report = train("resume-2-memory-cache-off", resumed)
    assert comparable(actual) == comparable(state)
    assert (work / "resume-2-memory-cache-off/model.json").read_bytes() == (work / "tiled/model.json").read_bytes()
    for key in ("final_training", "final_validation"):
        assert actual_report[key] == report[key]
    assert actual_report["learning_rate"]["first_update"] == 3
    assert actual_report["learning_rate"]["first_rate"] == 0.0025
    assert actual_report["cuda_gradient_profile"]["tiles"] == 13
    invalid = deepcopy(request)
    invalid["device"] = "cpu"
    train("reject-cpu", invalid, "requires an explicit CUDA device")
    invalid = deepcopy(resumed)
    invalid["config"]["cuda_tiling"]["max_frames"] = 2
    train("reject-capacity-change", invalid, "immutable configuration")
    invalid = deepcopy(resumed)
    invalid["config"].pop("cuda_tiling")
    train("reject-framewise-resume", invalid, "immutable configuration")
    receipt = {
        "schema": "omnimd-tiled-training-integration-v1", "status": "passed",
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_native_commit": "cce7f8c447b22824593ad3b4df333d63932e44bc",
        "integrated_native_commit": "0937cb97733d4a5bb5987389636e9ccd893fbe7b",
        "source_base": "6bb52955315304b1a35842e6065afc7219388805",
        "scope": "Short final engine CLI forwarding, same-model framewise equality and v7 restart on existing 24-frame original synthetic input; no new performance or capacity benchmark.",
        "reused_completed_successful_commands": reused,
        "fixture_atoms_per_frame_min_max": [atom_min, atom_max],
        "frames": 24, "training_frames": 18, "validation_frames": 6, "optimizer_updates": 7,
        "framewise_numeric_state_and_all_four_metrics_exact": True,
        "exported_model_bytes_exact": True,
        "v7_step_2_resume_memory_cache_off_exact_except_additional_steps_and_cache_flag": True,
        "tiling_plan": report["cuda_tiling"], "runs": runs,
        "rejected_before_output": ["CPU tiling", "changed tile capacity", "framewise resume of tiled checkpoint"],
        "source_sha256": report["source_sha256"],
        "sha256": {name: sha(path) for name, path in [
            ("engine", engine), ("trainer", trainer), ("script", Path(__file__)),
            ("helper", Path(__file__).with_name("validate_indexed_startup_integration.py")),
            ("dataset", work / "dataset.jsonl"), ("initial_model", work / "initial.json")]},
        "checks": [{"label": c["label"], "exit_code": c["exit_code"]} for c in commands]}
    write(work / "acceptance.json", receipt)
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
