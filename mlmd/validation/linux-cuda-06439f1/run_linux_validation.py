#!/usr/bin/env python3
"""Run the bounded candidate validation once, only after the owner's GPU probe.

Linux and Python standard library only. Three training subprocesses execute in
order: current v7 baseline, v8 GPU loss, v8 step-2 plus five-update resume.
Existing attempt markers are never reused. A command or comparison failure stops
the sequence; there is no retry. Exported models are subsequently evaluated on CPU.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value, exclusive=False):
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def difference(a, b):
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), "comparison object keys differ")
        return max((difference(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), "comparison list shapes differ")
        return max((difference(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        require(finite(a) and finite(b), "comparison contains nonfinite or nonnumeric data")
        return abs(a - b)
    require(type(a) is type(b) and a == b, "comparison nonnumeric metadata differs")
    return 0.0


def parameter_values(model):
    networks = [network for row in model["descriptor"]["embedding"] for network in row]
    networks += model["fitting"]
    return [value for network in networks for layer in network["layers"]
            for row in [*layer["weights"], layer["bias"], layer["timestep"] or []]
            for value in row]


def comparable_resume(state):
    state = deepcopy(state)
    state["config"].pop("steps")
    state["config"].pop("fixed_geometry_cache", None)
    return state


def comparison_metadata(state):
    state = deepcopy(state)
    for key in ("schema", "execution", "model", "first_moment", "second_moment"):
        state.pop(key)
    state["config"].pop("cuda_loss", None)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Owner authorization after the separate GPU probe")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--eval-exe", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    require(args.execute, "No workload started: the owner must pass --execute after the GPU probe")
    require(sys.platform.startswith("linux"), "This helper is for Linux only")
    require(math.isfinite(args.timeout) and args.timeout > 0, "timeout must be finite and positive")
    work = Path(__file__).resolve().parent
    manifest = read(work / "source-manifest.json")
    engine = (args.engine or work / "target/release/omnimd-mlmd").resolve()
    trainer = engine.with_name("omnimd-native-train")
    evaluator = (args.eval_exe or engine.with_name("omnimd-native-eval")).resolve()
    for label, path in (("engine", engine), ("trainer", trainer), ("CPU evaluator", evaluator)):
        require(path.is_file() and os.access(path, os.X_OK), f"missing executable {label}: {path}")
    for name, entry in manifest["inputs"].items():
        require(sha(work / name) == entry["sha256"], f"fixture hash differs: {name}")
    for name, entry in manifest["requests"].items():
        require(sha(work / name) == entry["sha256"], f"request hash differs: {name}")
        require(not (work / entry["output_directory"]).exists(), f"output already exists: {entry['output_directory']}")
    rows = [json.loads(line) for line in (work / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    frames = [row["data"]["frame"] for row in rows if row["kind"] == "frame"]
    require(len(frames) == 24, "fixture must contain 24 frames")
    require(sorted(set(len(frame["positions"]) for frame in frames)) == [3, 4], "fixture atom counts changed")
    frame_input = "".join(json.dumps(frame, allow_nan=False) + "\n" for frame in frames)
    attempts = work / "attempts"
    logs = work / "logs"
    attempts.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    require(not any(attempts.iterdir()), "Prior attempt exists; GPU/training retries are prohibited")
    run_info = {
        "status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
        "expected_candidate_commit": manifest["candidate_commit"],
        "candidate_provenance": manifest["build_provenance"],
        "sha256": {"engine": sha(engine), "trainer": sha(trainer), "cpu_evaluator": sha(evaluator),
                   "helper": sha(Path(__file__)), "manifest": sha(work / "source-manifest.json")},
        "inputs": manifest["inputs"], "requests": manifest["requests"],
        "training_environment": {key: os.environ.get(key) for key in
                                 ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "OMP_NUM_THREADS")},
    }
    write(work / "run-info.json", run_info, exclusive=True)
    commands = []
    active = None

    def execute(label, command, *, cpu=False, input_text=None):
        nonlocal active
        active = label
        marker = attempts / f"{label}.json"
        record = {"label": label, "command": [str(item) for item in command],
                  "started_utc": datetime.now(timezone.utc).isoformat(), "status": "starting",
                  "cpu_only": cpu, "exit_code": None, "duration_seconds": None}
        write(marker, record, exclusive=True)
        stdout_path, stderr_path = logs / f"{label}.stdout", logs / f"{label}.stderr"
        env = dict(os.environ)
        if cpu:
            env.update(CUDA_VISIBLE_DEVICES="-1", HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1")
        start = time.monotonic()
        stdout, stderr, process = "", "", None
        try:
            process = subprocess.Popen(record["command"], cwd=work, env=env, text=True,
                                       stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            record.update(status="running", pid=process.pid)
            write(marker, record)
            try:
                stdout, stderr = process.communicate(input=input_text, timeout=args.timeout)
                record["status"] = "passed" if process.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                record["status"] = "timed_out"
                # Terminate only this helper-created process group, including the
                # engine's native trainer child. Never leave a timed-out GPU child.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    stdout, stderr = process.communicate()
            record["exit_code"] = process.returncode
        except BaseException as error:
            record.update(status="launch_or_control_failed", error=str(error))
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
                record["exit_code"] = process.returncode
            raise
        finally:
            record["duration_seconds"] = time.monotonic() - start
            record["completed_utc"] = datetime.now(timezone.utc).isoformat()
            record.update(stdout=str(stdout_path.relative_to(work)), stderr=str(stderr_path.relative_to(work)))
            with stdout_path.open("x", encoding="utf-8") as stream:
                stream.write(stdout)
            with stderr_path.open("x", encoding="utf-8") as stream:
                stream.write(stderr)
            write(marker, record)
            commands.append(record)
            write(work / "commands.json", commands)
        print(json.dumps({"case": label, "status": record["status"], "exit_code": record["exit_code"],
                          "duration_seconds": record["duration_seconds"]}), flush=True)
        require(record["status"] == "passed", f"{label} failed; stop without retry; see {stderr_path}")
        return stdout

    states, reports = {}, {}
    def train(label):
        execute(label, [engine, "train-native", work / f"{label}.request.json"])
        state = read(work / label / "checkpoint.json")
        report = read(work / label / "report.json")
        states[label], reports[label] = state, report
        require(state["dataset_sha256"] == manifest["inputs"]["dataset.jsonl"]["sha256"], "checkpoint dataset differs")
        require(state["step"] == 7 and report["step_end"] == 7, "unexpected completed optimizer step")
        require(len(state["training_indices"]) == 18 and len(state["validation_indices"]) == 6, "unexpected split")
        require(report["native_v1_reload_max_abs_energy_force_virial"] == [0, 0, 0], "native export reload changed EFV")
        profile = report["cuda_gradient_profile"]
        for key, expected in manifest["expected_profiles"][label].items():
            require(profile[key] == expected, f"{label}: expected {key}={expected}, got {profile[key]}")
        require(profile["max_frames_per_launch"] == 3 and profile["max_atoms_per_launch"] <= 12, "unexpected actual tile shape")
        require(profile["peak_device_buffer_bytes"] <= 64 * 1024 * 1024, "device payload policy exceeded")
        require(profile["peak_host_tile_payload_bytes"] <= 32 * 1024 * 1024, "host tile payload policy exceeded")
        require(report["cuda_tiling"]["requested"] == {"max_frames": 3, "max_atoms": 16}, "tile capacities changed")
        gpu = label != "linux-v7"
        require(state["schema"] == f"omnimd-native-full-checkpoint-v{8 if gpu else 7}", "unexpected checkpoint schema")
        require(state["execution"]["backend"] == ("cuda_tiled_device_loss_f64_v1" if gpu else "cuda_tiled_reverse_f64_v1"), "unexpected executor")
        if gpu:
            require(profile["prediction_and_seeds_ms"] == 0.0, "GPU-loss mode performed host prediction/seeds")
        return state, report

    try:
        baseline, baseline_report = train("linux-v7")
        gpu, gpu_report = train("linux-v8")
        tolerance = manifest["parameter_moment_and_metric_absolute_tolerance"]
        parameter_delta = difference(parameter_values(gpu["model"]) + gpu["first_moment"] + gpu["second_moment"],
                                     parameter_values(baseline["model"]) + baseline["first_moment"] + baseline["second_moment"])
        require(parameter_delta < tolerance, "current v8/v7 parameters or moments exceed tolerance")
        require(comparison_metadata(gpu) == comparison_metadata(baseline), "v8/v7 non-executor metadata differs")
        require(gpu["execution"]["program_sha256"] != baseline["execution"]["program_sha256"], "v7/v8 program identity unexpectedly equal")
        metric_keys = ("initial_training", "final_training", "initial_validation", "final_validation")
        metric_deltas = {key: difference(gpu_report[key], baseline_report[key]) for key in metric_keys}
        require(max(metric_deltas.values()) < tolerance, "current v8/v7 metrics exceed tolerance")
        for key in ("minibatch", "learning_rate"):
            require(gpu_report[key] == baseline_report[key], f"v8/v7 {key} differs")
        require(gpu_report["minibatch"]["partial_batches"] == 2, "partial optimizer batches not exercised")
        resumed, resumed_report = train("linux-v8-resume")
        require(comparable_resume(resumed) == comparable_resume(gpu), "resume checkpoint differs beyond allowed steps/cache flags")
        for key in ("sampler", "model", "first_moment", "second_moment", "step", "execution"):
            require(resumed[key] == gpu[key], f"resume {key} is not exact")
        require((work / "linux-v8-resume/model.json").read_bytes() == (work / "linux-v8/model.json").read_bytes(), "resume exported model bytes differ")
        for key in ("final_training", "final_validation"):
            require(resumed_report[key] == gpu_report[key], f"resume {key} differs")
        require(resumed_report["step_start"] == 2 and resumed_report["learning_rate"]["first_update"] == 3, "resume update boundary changed")
        require(resumed_report["learning_rate"]["first_rate"] == 0.0025, "resume schedule boundary changed")

        evaluations = {}
        for label in ("linux-v7", "linux-v8", "linux-v8-resume"):
            raw = execute(label + "-cpu-export", [evaluator, work / label / "model.json"], cpu=True, input_text=frame_input)
            results = [json.loads(line) for line in raw.splitlines() if line.strip()]
            require(len(results) == len(frames), "CPU evaluator frame count differs")
            for frame, result in zip(frames, results):
                require(result["units"] == "EvAngstrom", "CPU evaluation units differ")
                require(finite(result["energy"]), "CPU energy is nonfinite")
                require(len(result["forces"]) == len(frame["positions"]), "CPU force atom count differs")
                for matrix, rows in ((result["forces"], len(frame["positions"])), (result["virial"], 3)):
                    require(len(matrix) == rows and all(len(row) == 3 and all(finite(x) for x in row) for row in matrix), "CPU EFV shape/nonfinite error")
            evaluations[label] = results
        efv_deltas = {}
        for label, reference in (("linux-v8", "linux-v7"), ("linux-v8-resume", "linux-v8")):
            delta = {key: max(difference(a[key], b[key]) for a, b in zip(evaluations[label], evaluations[reference]))
                     for key in ("energy", "forces", "virial")}
            limit = manifest["cpu_export_efv_absolute_tolerance"] if reference == "linux-v7" else 0.0
            require(max(delta.values()) <= limit, f"{label}: exported CPU EFV exceeds tolerance")
            efv_deltas[label + "_vs_" + reference] = delta
        receipt = {"status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "scope": manifest["scope"], "run_info": run_info, "commands": commands,
                   "current_v7_reference": "CPU prediction/seeds + CUDA reverse derivatives and Adam; current Linux candidate",
                   "current_v8_vs_v7_parameter_moment_max_abs": parameter_delta,
                   "current_v8_vs_v7_metric_max_abs": metric_deltas,
                   "comparison_absolute_tolerance": tolerance, "cpu_export_efv_max_abs": efv_deltas,
                   "resume_sampler_model_moments_step_execution_exact": True,
                   "resume_model_bytes_and_final_metrics_exact": True,
                   "runs": {label: {"schema": state["schema"], "execution": state["execution"],
                                     "profile": reports[label]["cuda_gradient_profile"],
                                     "learning_rate": reports[label]["learning_rate"],
                                     "model_sha256": sha(work / label / "model.json"),
                                     "checkpoint_sha256": sha(work / label / "checkpoint.json")}
                            for label, state in states.items()}}
        write(work / "acceptance.json", receipt, exclusive=True)
        print(json.dumps({"complete": True, "acceptance": str(work / "acceptance.json"),
                          "parameter_moment_max_abs": parameter_delta, "cpu_export_efv_max_abs": efv_deltas}), flush=True)
    except BaseException as error:
        write(work / "failure.json", {"status": "failed_no_retry", "active_or_last_command": active,
                                     "reason": str(error), "completed_utc": datetime.now(timezone.utc).isoformat(),
                                     "completed_command_records": commands}, exclusive=True)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Validation stopped without retry: {error}", file=sys.stderr)
        sys.exit(1)
