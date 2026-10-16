"""Compare instrumented duplicate-validation baseline with the consolidated trainer.
CPU only; Python stdlib only. Reuse the original indexed synthetic replay input.
Both executables must expose the same startup timing instrumentation.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(65536), b""):
            h.update(part)
    return h.hexdigest()


def comparable(state):
    state = deepcopy(state)
    state["config"].pop("steps", None)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-exe", type=Path, required=True)
    parser.add_argument("--candidate-exe", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three repeats are required")
    fixture = args.fixture_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    executables = {"baseline": args.baseline_exe.resolve(), "candidate": args.candidate_exe.resolve()}
    template = read(fixture / "large.request.json")
    previous = read(fixture / "large/checkpoint.json")
    previous_midpoint = read(fixture / "large/checkpoint-step-16.json")
    dataset = fixture / "large.jsonl"
    frames = len(previous["training_indices"]) + len(previous["validation_indices"])
    samples = {case: {version: [] for version in executables} for case in ("fresh", "resume")}
    metrics = {}
    checkpoints = {}
    checkpoint_hashes = {}
    for repeat in range(args.repeats):
        for case in samples:
            # Alternate execution order without changing dataset or training order.
            order = ("baseline", "candidate") if (repeat + (case == "resume")) % 2 == 0 else ("candidate", "baseline")
            for version in order:
                name = f"{case}-{version}-{repeat + 1}"
                request = deepcopy(template)
                request["dataset"] = str(dataset)
                request["output_directory"] = name
                if case == "resume":
                    request.pop("initial_model")
                    request["checkpoint"] = str(fixture / "large/checkpoint-step-16.json")
                    request["config"]["steps"] = 3
                else:
                    request["initial_model"] = str(fixture / "initial.json")
                path = out / (name + ".request.json")
                write(path, request)
                start = time.perf_counter()
                proc = subprocess.run([str(executables[version]), str(path)], capture_output=True, text=True)
                elapsed = 1000 * (time.perf_counter() - start)
                (out / (name + ".stdout.txt")).write_text(proc.stdout, encoding="utf-8")
                (out / (name + ".stderr.txt")).write_text(proc.stderr, encoding="utf-8")
                if proc.returncode:
                    raise RuntimeError(f"{name}: {proc.stderr}")
                state = read(out / name / "checkpoint.json")
                report = read(out / name / "report.json")
                model = read(out / name / "model.json")
                assert comparable(state) == comparable(previous), name
                assert model == state["model"], name
                if case == "fresh":
                    assert read(out / name / "checkpoint-step-16.json") == previous_midpoint, name
                selected_metrics = {key: report[key] for key in (
                    "initial_training", "final_training", "initial_validation", "final_validation",
                    "native_v1_reload_max_abs_energy_force_virial", "minibatch"
                )}
                if case in metrics:
                    assert selected_metrics == metrics[case], name
                    assert state == checkpoints[case], name
                else:
                    metrics[case] = selected_metrics
                    checkpoints[case] = state
                    checkpoint_hashes[case] = digest(out / name / "checkpoint.json")
                access = report["dataset_access"]
                timing = report["startup_timing"]
                assert access["frames"] == frames
                assert report["dataset_sha256"] == previous["dataset_sha256"]
                sample = {
                    **timing,
                    "trainer_startup_ms": report["timing"]["startup_ms"],
                    "trainer_validation_and_initialization_ms": report["timing"]["initialization_ms"],
                    "process_ms": elapsed,
                    "frame_read_ms": access["frame_read_ms"],
                    "integrity_verification_ms": access["integrity_verification_ms"],
                    "startup_frame_reads": report["timing"]["startup_frame_reads"],
                    "frame_reads": access["frame_reads"],
                    "integrity_verifications": access["integrity_verifications"],
                    "run": name,
                }
                samples[case][version].append(sample)
                print(f"{name}: startup={timing['before_first_update_ms']:.3f}ms process={elapsed:.3f}ms reads={access['frame_reads']}", flush=True)
    results = {}
    for case, versions in samples.items():
        baseline, candidate = versions["baseline"], versions["candidate"]
        for old, new in zip(baseline, candidate):
            assert old["startup_frame_reads"] - new["startup_frame_reads"] == 2 * frames
            assert old["frame_reads"] - new["frame_reads"] == 2 * frames
            assert old["integrity_verifications"] == new["integrity_verifications"]
        medians = {}
        for version, runs in versions.items():
            medians[version] = {
                key: statistics.median(run[key] for run in runs)
                for key in runs[0] if key.endswith("_ms")
            }
        results[case] = {
            "samples": versions,
            "median_ms": medians,
            "startup_reduction_percent": 100 * (1 - medians["candidate"]["before_first_update_ms"] / medians["baseline"]["before_first_update_ms"]),
            "process_reduction_percent": 100 * (1 - medians["candidate"]["process_ms"] / medians["baseline"]["process_ms"]),
            "removed_frame_reads": 2 * frames,
            "all_checkpoint_fields_model_metrics_sampler_exact": True,
            "historical_final_checkpoint_exact_except_requested_steps": True,
            "whole_file_verification_count_unchanged": True,
            "checkpoint_sha256": checkpoint_hashes[case],
        }
    rejected = []
    for case in ("zero_steps", "cpu_geometry_cache", "checkpoint_moments"):
        request = deepcopy(template)
        request["dataset"] = str(dataset)
        request["initial_model"] = str(fixture / "initial.json")
        request["output_directory"] = "invalid-" + case
        if case == "zero_steps":
            request["config"]["steps"] = 0
        elif case == "cpu_geometry_cache":
            request["config"]["fixed_geometry_cache"] = True
        else:
            bad = deepcopy(previous_midpoint)
            bad["first_moment"].pop()
            checkpoint_path = out / "invalid-checkpoint.json"
            write(checkpoint_path, bad)
            request.pop("initial_model")
            request["checkpoint"] = str(checkpoint_path)
        path = out / ("invalid-" + case + ".request.json")
        write(path, request)
        proc = subprocess.run([str(executables["candidate"]), str(path)], capture_output=True, text=True)
        assert proc.returncode != 0, case
        assert not (out / request["output_directory"]).exists(), case
        rejected.append({"case": case, "nonzero_exit_and_no_output_directory": True, "diagnostic": proc.stderr.strip()})
    receipt = {
        "scope": "CPU, original synthetic 2051-frame whitespace-padded input; repeated warm filesystem-cache comparison; no GPU or physical/scaling claim",
        "frames": frames,
        "file_bytes": dataset.stat().st_size,
        "repeats_per_case_per_executable": args.repeats,
        "order": "alternating baseline/candidate within each fresh/resume case",
        "timing_boundaries": {
            "dataset_open_ms": "complete initial JSONL scan, strict parse and index construction",
            "state_preparation_ms": "initial model parse plus constructor validation, or checkpoint parse",
            "before_trainer_ms": "CLI run entry to trainer entry; baseline includes redundant CLI preflight",
            "trainer_startup_ms": "trainer entry through validation, initialization and initial train/validation assessments, before first update",
            "before_first_update_ms": "before_trainer_ms + trainer_startup_ms; excludes process launch",
            "process_ms": "Python subprocess launch through completion, all updates, assessments and exports",
            "startup_frame_reads": "cumulative indexed frame reads through initial assessments; index scan is counted separately",
        },
        "results": results,
        "invalid_requests": rejected,
        "sha256": {
            "dataset": digest(dataset),
            "initial_model": digest(fixture / "initial.json"),
            "resume_checkpoint": digest(fixture / "large/checkpoint-step-16.json"),
            "script": digest(Path(__file__).resolve()),
            **{version + "_executable": digest(exe) for version, exe in executables.items()},
        },
    }
    write(out / "acceptance.json", receipt)
    print(json.dumps({case: {key: value for key, value in result.items() if key not in ("samples", "median_ms")} for case, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()
