"""Compare packed-transfer and historical trainers on identical prepared requests.
Python stdlib only; run sequentially in an authorized exclusive GPU window.
Each case directory contains initial.json, dataset.jsonl, profile.request.json.
No inputs or labels are synthesized or rescaled by this script.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
from replay_full_training import digest, write
from replay_indexed_training import launch
from replay_cuda_reverse import values


def numeric_error(a, b):
    aa = values(a["model"]) + a["first_moment"] + a["second_moment"]
    bb = values(b["model"]) + b["first_moment"] + b["second_moment"]
    assert len(aa) == len(bb)
    return max(abs(x-y) for x, y in zip(aa, bb))


def comparable(s):
    s = deepcopy(s)
    s["config"].pop("steps", None)
    s["config"].pop("fixed_geometry_cache", None)
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before-exe", type=Path, required=True)
    p.add_argument("--after-exe", type=Path, required=True)
    p.add_argument("--prepared-dir", type=Path, required=True)
    p.add_argument("--output-name", required=True)
    p.add_argument("--cases", nargs="+", default=["physical", "medium", "tiny"])
    p.add_argument("--cpu", action="store_true", help="Also run one identical-work CPU AD reference per case")
    a = p.parse_args()
    root = a.prepared_dir.resolve()
    out = root / a.output_name
    out.mkdir(exist_ok=False)
    before, after = a.before_exe.resolve(), a.after_exe.resolve()
    result = {"scope": "FP64 transfer batching only; unchanged work, kernels, and loss tolerances",
              "measurement": "Three alternating-order fresh processes per GPU implementation; startup includes NVRTC/driver initialization with uncontrolled driver cache; steady_updates_ms = gradient_ms + adam_ms (excludes assessment/IO)",
              "sha256": {"before_exe": digest(before), "after_exe": digest(after), "script": digest(__file__)}, "cases": []}
    for name in a.cases:
        folder = root / name
        req = json.loads((folder / "profile.request.json").read_text())
        req["initial_model"] = str(folder / "initial.json")
        req["dataset"] = str(folder / "dataset.jsonl")
        req["config"]["checkpoint_every"] = req["config"]["steps"]
        runs = []
        def execute(label, exe, request):
            request = deepcopy(request)
            request["output_directory"] = str(out / (name + "-" + label))
            path = out / (name + "-" + label + ".request.json")
            write(path, request)
            proc = launch(exe, path)
            destination = Path(request["output_directory"])
            state = json.loads((destination / "checkpoint.json").read_text())
            report = json.loads((destination / "report.json").read_text())
            assert report["native_v1_reload_max_abs_energy_force_virial"] == [0., 0., 0.]
            row = {"label": label, "process": proc, "timing": report["timing"],
                   "profile": report.get("cuda_gradient_profile"),
                   "checkpoint_sha256": digest(destination / "checkpoint.json")}
            runs.append(row)
            print(json.dumps({"case": name, "run": label, **proc, "timing": report["timing"]}), flush=True)
            write(out / "progress.json", {"completed_cases": result["cases"], "active_case": name, "runs": runs})
            return state, report
        states, reports = {}, {}
        for repeat in range(3):
            for mode in (("before", "after") if repeat % 2 == 0 else ("after", "before")):
                state, report = execute(f"{mode}-{repeat}", before if mode == "before" else after, req)
                if states:
                    assert state == next(iter(states.values())), "checkpoint numerical identity changed"
                    reference = next(iter(reports.values()))
                    assert report["execution"] == reference["execution"], "PTX/execution identity changed"
                    for key in ("final_training", "final_validation", "training_frames", "validation_frames"):
                        assert report[key] == reference[key], key
                    assert report["cuda_tiling"]["effective_max_atoms"] == reference["cuda_tiling"]["effective_max_atoms"]
                    for key in ("frames", "tiles", "kernel_launches", "synchronizations"):
                        assert report["cuda_gradient_profile"][key] == reference["cuda_gradient_profile"][key]
                if mode == "after":
                    prof = report["cuda_gradient_profile"]
                    assert prof["input_transfer_calls"] == 2 * prof["tiles"]
                    assert prof["peak_device_buffer_bytes"] <= 64*1024*1024
                    assert prof["peak_host_tile_payload_bytes"] <= 32*1024*1024
                states[mode], reports[mode] = state, report
        medians = {}
        for mode in ("before", "after"):
            rows = [r for r in runs if r["label"].startswith(mode)]
            medians[mode] = {"whole_process_ms": statistics.median(r["process"]["process_ms"] for r in rows),
                "startup_ms": statistics.median(r["timing"]["startup_ms"] for r in rows),
                "steady_updates_ms": statistics.median(r["timing"]["gradient_ms"]+r["timing"]["adam_ms"] for r in rows),
                "gradient_ms": statistics.median(r["timing"]["gradient_ms"] for r in rows),
                "upload_ms": statistics.median(r["profile"]["upload_ms"] for r in rows)}
        case = {"name": name, "dataset_sha256": digest(folder / "dataset.jsonl"), "initial_model_sha256": digest(folder / "initial.json"),
                "steps": req["config"]["steps"], "request": req, "medians": medians,
                "whole_speed_ratio": medians["before"]["whole_process_ms"] / medians["after"]["whole_process_ms"],
                "steady_speed_ratio": medians["before"]["steady_updates_ms"] / medians["after"]["steady_updates_ms"],
                "exact_checkpoint_model_moments_sampler_metrics_and_ptx": True, "execution": reports["after"]["execution"], "runs": runs}
        if name == "physical":
            half = deepcopy(req); half["config"]["steps"] //= 2
            old_half, _ = execute("old-half", before, half)
            resume = deepcopy(half); resume.pop("initial_model")
            resume["checkpoint"] = str(out / (name + "-old-half") / "checkpoint.json")
            resume["config"]["fixed_geometry_cache"] = False
            resume["dataset_access"] = "indexed"
            resumed, rr = execute("new-resume", after, resume)
            assert comparable(resumed) == comparable(states["after"])
            assert rr["final_training"] == reports["after"]["final_training"]
            case["cross_binary_resume_cache_and_storage_switch_exact"] = True
        if a.cpu:
            cpu_req = deepcopy(req); cpu_req["device"] = "cpu"
            for key in ("cuda_gradient", "cuda_loss", "cuda_tiling", "fixed_geometry_cache"):
                cpu_req["config"].pop(key, None)
            cs, cr = execute("cpu-reference", after, cpu_req)
            error = numeric_error(cs, states["after"])
            assert error < 1e-10, error
            assert cs["sampler"] == states["after"]["sampler"]
            case["cpu_reference"] = {"model_moments_max_abs": error, "same_sampler": True,
                "whole_speed_ratio": runs[-1]["process"]["process_ms"] / medians["after"]["whole_process_ms"],
                "steady_speed_ratio": (cr["timing"]["gradient_ms"]+cr["timing"]["adam_ms"]) / medians["after"]["steady_updates_ms"],
                "final_training": cr["final_training"], "final_validation": cr["final_validation"]}
        result["cases"].append(case)
        write(out / "acceptance.json", result)
        print(json.dumps({"case": name, "whole_speed_ratio": case["whole_speed_ratio"], "steady_speed_ratio": case["steady_speed_ratio"], "cpu_reference": case.get("cpu_reference")}), flush=True)
    print("Transfer replay completed", flush=True)


if __name__ == "__main__":
    main()
