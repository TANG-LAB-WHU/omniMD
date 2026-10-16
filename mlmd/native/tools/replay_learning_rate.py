"""Focused native learning-rate CLI replay using the original synthetic minibatch fixture.
CPU or explicit CUDA only. No physical-accuracy or DeepMD-default equivalence claim.
Generate the fixture with replay_minibatch_training.py; Python stdlib only.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time
from replay_minibatch_training import values


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparable(state):
    state = deepcopy(state)
    state["config"].pop("steps", None)
    state["config"].pop("fixed_geometry_cache", None)
    return state


def numerical(state):
    return values(state["model"]) + state["first_moment"] + state["second_moment"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-exe", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cuda-gradient", choices=("hyperdual", "reverse"), default="hyperdual")
    parser.add_argument("--cpu-reference", type=Path)
    args = parser.parse_args()
    if args.device != "cpu" and not (args.device.startswith("cuda:") and args.device[5:].isascii() and args.device[5:].isdecimal()):
        raise ValueError("device must be cpu or explicit cuda:N")
    if args.device == "cpu" and args.cuda_gradient != "hyperdual":
        raise ValueError("reverse requires CUDA")
    fixture = args.fixture_dir.resolve()
    trainer = args.train_exe.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    template = read(fixture / "continuous.request.json")
    template.update(device=args.device, dataset=str(fixture / "dataset.jsonl"),
                    initial_model=str(fixture / "initial.json"), dataset_access="indexed")
    template["config"].update(steps=7, checkpoint_every=1)
    if args.cuda_gradient == "reverse":
        template["config"]["cuda_gradient"] = "reverse"
    if args.device != "cpu":
        template["config"]["fixed_geometry_cache"] = True
    schedule = {"kind": "exponential", "decay_steps": 2, "decay_rate": 0.5}
    template["config"]["learning_rate_schedule"] = schedule
    runs = {}

    def execute(name, request):
        request = deepcopy(request)
        request["output_directory"] = name
        path = out / (name + ".request.json")
        write(path, request)
        start = time.perf_counter()
        proc = subprocess.run([str(trainer), str(path)], capture_output=True, text=True)
        elapsed = (time.perf_counter() - start) * 1000
        (out / (name + ".stdout.txt")).write_text(proc.stdout, encoding="utf-8")
        (out / (name + ".stderr.txt")).write_text(proc.stderr, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(f"{name}: {proc.stderr}")
        state = read(out / name / "checkpoint.json")
        report = read(out / name / "report.json")
        rates = report["learning_rate"]
        mode = request["config"].get("learning_rate_schedule") or {"kind": "constant"}
        start_step, end_step = report["step_start"], report["step_end"]
        def expected(k):
            return request["config"]["learning_rate"] * (mode["decay_rate"] ** (k // mode["decay_steps"]) if mode["kind"] == "exponential" else 1)
        assert rates["contract"] == "omnimd-optimizer-update-learning-rate-v1"
        assert rates["schedule"] == mode
        assert rates["first_update"] == start_step + 1 and rates["last_update"] == end_step
        assert rates["first_rate"] == expected(start_step) and rates["last_rate"] == expected(end_step - 1)
        assert report["native_v1_reload_max_abs_energy_force_virial"] == [0.0, 0.0, 0.0]
        runs[name] = {"process_ms": elapsed, "rates": rates, "checkpoint_sha256": digest(out / name / "checkpoint.json")}
        print(f"{name}: steps {start_step}->{end_step}, rates {rates['first_rate']}->{rates['last_rate']}", flush=True)
        return state, report

    state, report = execute("scheduled", template)
    assert state["schema"] == "omnimd-native-full-checkpoint-v6"
    expected_rates = [0.005, 0.005, 0.0025, 0.0025, 0.00125, 0.00125, 0.000625]
    assert report["learning_rate"]["first_rate"] == expected_rates[0]
    assert report["learning_rate"]["last_rate"] == expected_rates[-1]
    assert report["minibatch"]["partial_batches"] == 2
    memory = deepcopy(template)
    memory["dataset_access"] = "memory"
    memory_state, _ = execute("scheduled-memory", memory)
    assert comparable(state) == comparable(memory_state)
    for split in (1, 2, 3, 4):
        request = deepcopy(template)
        request.pop("initial_model")
        request["checkpoint"] = str(out / f"scheduled/checkpoint-step-{split}.json")
        request["config"]["steps"] = 7 - split
        if split % 2 == 0:
            request["dataset_access"] = "memory"
        if args.device != "cpu":
            request["config"]["fixed_geometry_cache"] = False
        resumed, resumed_report = execute(f"resume-{split}", request)
        assert comparable(state) == comparable(resumed)
        for key in ("final_training", "final_validation"):
            assert report[key] == resumed_report[key]
        assert resumed_report["learning_rate"]["first_rate"] == expected_rates[split]

    full = deepcopy(template)
    full["config"].pop("minibatch")
    full["dataset_access"] = "memory"
    full_state, full_report = execute("full-batch", full)
    for split in (2, 3):
        request = deepcopy(full)
        request.pop("initial_model")
        request["checkpoint"] = str(out / f"full-batch/checkpoint-step-{split}.json")
        request["config"]["steps"] = 7 - split
        resumed, resumed_report = execute(f"full-batch-resume-{split}", request)
        assert comparable(full_state) == comparable(resumed)
        assert full_report["final_training"] == resumed_report["final_training"]

    legacy = deepcopy(template)
    legacy["config"].pop("learning_rate_schedule")
    legacy_state, _ = execute("legacy", legacy)
    assert legacy_state["schema"] == "omnimd-native-full-checkpoint-v5"
    constant = deepcopy(legacy)
    constant["config"]["learning_rate_schedule"] = {"kind": "constant"}
    constant_state, _ = execute("constant", constant)
    assert constant_state["schema"] == "omnimd-native-full-checkpoint-v6"
    assert numerical(legacy_state) == numerical(constant_state)
    unity = deepcopy(template)
    unity["config"]["learning_rate_schedule"]["decay_rate"] = 1.0
    unity_state, _ = execute("unity-decay", unity)
    assert numerical(legacy_state) == numerical(unity_state)
    historical = None
    if args.device == "cpu":
        historical_path = fixture / "continuous/checkpoint-step-7.json"
        assert comparable(legacy_state) == comparable(read(historical_path))
        historical = {"checkpoint_sha256": digest(historical_path), "all_fields_exact_except_requested_steps": True}

    failures = []
    for case in ("zero_interval", "rate_above_one", "zero_rate", "unknown_field", "unknown_kind", "underflow", "changed_schedule", "removed_schedule", "changed_base"):
        request = deepcopy(template)
        cfg = request["config"]
        if case == "zero_interval":
            cfg["learning_rate_schedule"]["decay_steps"] = 0
        elif case == "rate_above_one":
            cfg["learning_rate_schedule"]["decay_rate"] = 1.1
        elif case == "zero_rate":
            cfg["learning_rate_schedule"]["decay_rate"] = 0.0
        elif case == "unknown_field":
            cfg["learning_rate_schedule"] = {"kind": "constant", "decay_rate": 0.5}
        elif case == "unknown_kind":
            cfg["learning_rate_schedule"] = {"kind": "cosine"}
        elif case == "underflow":
            cfg["learning_rate"] = 1.0
            cfg["learning_rate_schedule"]["decay_steps"] = 1
            cfg["steps"] = 1076
        else:
            request.pop("initial_model")
            request["checkpoint"] = str(out / "scheduled/checkpoint-step-2.json")
            if case == "changed_schedule":
                cfg["learning_rate_schedule"]["decay_steps"] = 3
            elif case == "removed_schedule":
                cfg.pop("learning_rate_schedule")
            else:
                cfg["learning_rate"] = 0.01
        request["output_directory"] = "invalid-" + case
        path = out / ("invalid-" + case + ".request.json")
        write(path, request)
        proc = subprocess.run([str(trainer), str(path)], capture_output=True, text=True)
        assert proc.returncode != 0 and not (out / request["output_directory"]).exists(), case
        failures.append({"case": case, "rejected_before_output_creation": True, "diagnostic": proc.stderr.strip()})

    comparison = None
    if args.cpu_reference:
        reference = args.cpu_reference.resolve()
        comparison = {}
        for name, actual in (("scheduled", state), ("full-batch", full_state)):
            cpu = read(reference / name / "checkpoint.json")
            difference = max(abs(a - b) for a, b in zip(numerical(actual), numerical(cpu)))
            assert difference < 3e-12, (name, difference)
            assert len(numerical(actual)) == len(numerical(cpu))
            assert actual["sampler"] == cpu["sampler"] if "sampler" in actual else "sampler" not in cpu
            comparison[name + "_max_abs_parameters_and_moments"] = difference
    receipt = {
        "scope": "original synthetic eight-frame E/F/V fixture; seven optimizer updates; no physical, speed or DeepMD-default equivalence claim",
        "device": args.device, "gradient": args.cuda_gradient, "execution": state.get("execution", {"backend": "cpu_reference_v1"}),
        "schedule": schedule, "expected_update_rates": expected_rates,
        "indexed_memory_exact": True, "minibatch_resume_splits_exact": [1, 2, 3, 4],
        "full_batch_resume_splits_exact": [2, 3], "cache_toggle_on_cuda_resume": args.device != "cpu",
        "constant_and_unity_decay_match_legacy_parameters_and_moments_exactly": True,
        "historical_cpu_checkpoint": historical, "cpu_reference_comparison": comparison,
        "runs": runs, "invalid_requests": failures,
        "sha256": {"trainer": digest(trainer), "dataset": digest(fixture / "dataset.jsonl"),
                   "initial_model": digest(fixture / "initial.json"), "script": digest(Path(__file__).resolve())},
    }
    write(out / "acceptance.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key not in ("runs", "invalid_requests")}, indent=2))


if __name__ == "__main__":
    main()
