"""Minimal engine forwarding and historical-resume acceptance; not a benchmark.
Reuse B's accepted resident-state work and requests. Run only in an exclusive GPU
window. Stops on the first failure without retry; never changes reference inputs.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--reference-work", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    a = parser.parse_args()
    engine, reference, work = a.engine.resolve(), a.reference_work.resolve(), a.work.resolve()
    trainer = engine.with_name("omnimd-native-train.exe" if os.name == "nt" else "omnimd-native-train")
    assert engine.is_file() and trainer.is_file()
    work.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    receipt_path = root/"mlmd/native/fixtures/resident-state-report.json"
    receipt = read(receipt_path)
    physical = next(c for c in receipt["cases"] if c["case"] == "physical")
    for name, expected in receipt["source_sha256_lf_normalized"].items():
        data = (root/"mlmd/native"/name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == expected, name
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")

    def launch(name, arguments):
        result = subprocess.run([str(engine), *map(str, arguments)], env=env,
                                capture_output=True, text=True, timeout=120)
        (work/(name+".stdout.log")).write_text(result.stdout, encoding="utf-8")
        (work/(name+".stderr.log")).write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{name} failed ({result.returncode}); no retry: {result.stderr}")
        return result.stdout

    caps = json.loads(launch("capabilities", ["capabilities"]))
    assert caps["backends"]["native"]["default_device"] == "Cpu"
    assert caps["backends"]["native"]["cuda_compiled"]
    assert "CPU default" in caps["training"]["backend"]
    assert "state_residency" in caps["training"]["cuda"]["loss_preparation"]
    summary = {
        "schema": "omnimd-resident-state-integrated-v1",
        "scope": "Actual engine train-native forwarding: one unchanged public 1000-update request and one historical v8 500-update continuation; no performance benchmark",
        "baseline_commit": "201cdef0626bd9308bcff07faef4c5ae3aa9d93a",
        "native_source_commit": "f4ca7f3a6ec59e7164c6482d574eb7ca650b2a9a",
        "sha256": {"engine": digest(engine), "trainer": digest(trainer), "script": digest(__file__), "native_receipt": digest(receipt_path)},
        "cpu_default_and_explicit_gpu_capability": True,
        "reference_work": str(reference), "local_artifact_root": str(work), "cases": [],
    }
    for label, source, expected_steps in [
        ("public-full", "physical-current-0", 1000),
        ("historical-v8-resume", "historical-v8-resume", 500),
    ]:
        original = read(reference/(source+".request.json"))
        request = deepcopy(original)
        assert request["device"] == "cuda:0" and request["config"]["cuda_loss"] == "gpu"
        assert request["config"]["steps"] == expected_steps
        assert digest(request["dataset"]) == physical["dataset_sha256"]
        if "initial_model" in request:
            assert digest(request["initial_model"]) == physical["initial_model_sha256"]
            assert request["config"] == physical["config"]
        request["output_directory"] = str(work/label)
        assert {k: v for k, v in request.items() if k != "output_directory"} == {
            k: v for k, v in original.items() if k != "output_directory"}
        ip = work/(label+".request.json")
        write(ip, request)
        launch(label, ["train-native", ip])
        actual, expected = work/label, reference/source
        report, old_report = read(actual/"report.json"), read(expected/"report.json")
        for name in ("checkpoint.json", "checkpoint-step-1000.json", "model.json"):
            assert (actual/name).read_bytes() == (expected/name).read_bytes(), (label, name)
        assert digest(actual/"model.json") == physical["identical_gpu_files"]["model_sha256"]
        for key in ("execution", "final_training", "final_validation", "training_frames", "validation_frames", "minibatch", "cuda_tiling"):
            assert report[key] == old_report[key], (label, key)
        assert report["cuda_gradient_profile"] == {
            **old_report["cuda_gradient_profile"],
            **{k: v for k, v in report["cuda_gradient_profile"].items() if k.endswith("_ms")}
        }, label
        resident = report["cuda_state_residency"]
        for key, value in {"updates": expected_steps, "status_downloads": expected_steps,
                           "snapshots": 1, "state_download_calls": 3, "state_download_bytes": 12744}.items():
            assert resident[key] == value, (label, key)
        assert report["native_v1_reload_max_abs_energy_force_virial"] == [0., 0., 0.]
        assert report["output_timing"]["reload_checked_frames"] == 211
        assert report["output_timing"]["reload_evaluations"] == 422
        summary["cases"].append({
            "case": label, "updates": expected_steps, "final_step": report["step_end"],
            "request_only_output_path_changed": True,
            "checkpoint_periodic_checkpoint_and_model_bytes_match_reference": True,
            "execution_metrics_sampler_and_tile_counts_match_reference": True,
            "resident_state": {k: v for k, v in resident.items() if not k.endswith("_ms")},
            "all_dataset_export_evaluations": 422, "reload_max_abs_energy_force_virial": [0., 0., 0.],
            "model_sha256": digest(actual/"model.json"), "checkpoint_sha256": digest(actual/"checkpoint.json"),
            "request_sha256": digest(ip), "reference_request_sha256": digest(reference/(source+".request.json")),
        })
        write(work/"acceptance.json", summary)
        print(label+": exact artifact and metric agreement; resident-state counts passed", flush=True)
    assert (work/"public-full/model.json").read_bytes() == (work/"historical-v8-resume/model.json").read_bytes()
    summary.update(status="passed", gpu_runs=2, gpu_failures=0, retries=0,
                   historical_resume_matches_full_model=True,
                   performance_measurement=False, gpu_window_released=True)
    write(work/"acceptance.json", summary)
    print("Minimal resident-state integration acceptance passed; GPU work complete.", flush=True)


if __name__ == "__main__":
    main()
