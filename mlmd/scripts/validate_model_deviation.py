"""CPU ensemble acceptance using original synthetic models and Python stdlib oracles.
No download, build, GPU use or overwrite. Generated models are software fixtures.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--eval-exe", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    engine, evaluator, work = args.engine.resolve(), args.eval_exe.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    commands = []

    def run(label, command, text=None, success=True):
        command = list(map(str, command))
        result = subprocess.run(command, input=text, encoding="utf-8", capture_output=True, timeout=120)
        number = len(commands) + 1
        (work / f"{number:02d}.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (work / f"{number:02d}.stderr.txt").write_text(result.stderr, encoding="utf-8")
        commands.append({"label": label, "command": command, "exit_code": result.returncode})
        print(f"{number}. {label}: {result.returncode}", flush=True)
        if (result.returncode == 0) != success:
            raise AssertionError(result.stderr)
        return result.stdout

    def request(name, value):
        path = work / (name + ".request.json")
        write(path, value)
        return path

    base = read(source / "native/fixtures/frame.jsonl")
    frames = []
    rng = random.Random(20261016)
    for index in range(12):
        frame = deepcopy(base)
        if index % 2 == 0:
            frame["cell"] = None
        for position in frame["positions"]:
            for axis in range(3):
                position[axis] += rng.uniform(-0.15, 0.15)
        if index % 3 == 0:
            frame["positions"].pop()
            frame["species"].pop()
        frames.append(frame)
    frame_text = "".join(json.dumps(frame) + "\n" for frame in frames)
    members, evaluations = [], []
    template = read(source / "native/fixtures/synthetic-v1.json")
    for member in range(3):
        model = deepcopy(template)
        for kind, net in enumerate(model["fitting"]):
            for index, row in enumerate(net["layers"][-1]["weights"]):
                row[0] += 0.4 * math.sin(member * 1.3 + kind * 0.7 + index)
        path = work / f"model-{member}.json"
        write(path, model)
        members.append({"id": f"member-{member}", "options": {"native_model": str(path), "device": "cpu"}})
        evaluations.append([json.loads(line) for line in run(f"Independent model {member}", [evaluator, path], frame_text).splitlines()])
    header = {"schema": "omnimd-dataset-jsonl-v1", "type_map": template["type_map"], "units": "EvAngstrom",
              "virial_convention": "MinusDEpsilonRow", "source": "original synthetic model-deviation replay", "selected_sets": []}
    records = [{"kind": "header", "header": header}]
    for index, frame in enumerate(frames):
        labels = {key: evaluations[0][index][key] if index % 2 else None for key in ("energy", "forces", "virial")}
        records.append({"kind": "frame", "data": {"index": index, "id": f"deviation/{index}", "frame": frame, "labels": labels}})
    records.append({"kind": "completed", "frames": len(frames), "source_sha256": {"generator": digest(__file__)}})

    def dataset(name, values):
        path = work / name
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for value in values:
                stream.write(json.dumps(value, allow_nan=False) + "\n")
        return path

    data = dataset("dataset.jsonl", records)
    expected = []
    for index, frame in enumerate(frames):
        n = len(frame["positions"])
        values = [model[index] for model in evaluations]
        energy = [v["energy"] / n for v in values]
        force = [math.hypot(*(statistics.pstdev(v["forces"][a][k] for v in values) for k in range(3))) for a in range(n)]
        virial = [[statistics.pstdev(v["virial"][a][b] / n for v in values) for b in range(3)] for a in range(3)]
        flat = sum(virial, [])
        expected.append({"energy_per_atom_std_ev": statistics.pstdev(energy), "mean_energy_per_atom_ev": statistics.fmean(energy),
            "force_ev_per_angstrom": {"min": min(force), "max": max(force), "mean": statistics.fmean(force), "per_atom": force},
            "virial_per_atom_ev": {"components": virial, "min": min(flat), "max": max(flat), "rms": math.hypot(*flat) / 3}})
    ordered = sorted(s["force_ev_per_angstrom"]["max"] for s in expected)
    lower, upper = (ordered[2] + ordered[3]) / 2, (ordered[8] + ordered[9]) / 2
    req = {"schema": "omnimd-model-deviation-v1", "models": members, "dataset": str(data), "per_atom": True,
           "force_window": {"min_ev_per_angstrom": lower, "max_ev_per_angstrom": upper}}
    output = work / "deviations.jsonl"
    run("CPU ensemble", [engine, "model-deviation", request("ensemble", req), output])
    actual = lines(output)
    assert actual[-1]["kind"] == "completed"
    assert actual[-1]["dataset_sha256"] == digest(data)
    assert actual[-1]["declared_source_sha256"] == records[-1]["source_sha256"]
    assert actual[-1]["combined_evaluations"] == 36 and actual[-1]["model_loads"] == 3
    assert actual[0]["distinct_model_byte_hashes"] == 3
    max_error = 0.0

    def compare(a, b):
        nonlocal max_error
        if isinstance(b, dict):
            assert set(a) == set(b)
            for key in b:
                compare(a[key], b[key])
        elif isinstance(b, list):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                compare(x, y)
        else:
            max_error = max(max_error, abs(a - b))
            assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-13), (a, b)
    candidates = []
    for index, (entry, oracle) in enumerate(zip(actual[1:-1], expected)):
        assert entry["index"] == index and entry["id"] == f"deviation/{index}"
        compare(entry["statistics"], oracle)
        selected = lower <= oracle["force_ev_per_angstrom"]["max"] < upper
        assert entry["candidate"] == selected
        if selected:
            candidates.append(index)
    assert len(candidates) == actual[-1]["candidate_frames"] == 6
    # Labels are validated, but their numerical values never enter disagreement.
    altered = deepcopy(records)
    for entry in altered[1:-1]:
        n = len(entry["data"]["frame"]["species"])
        entry["data"]["labels"] = {"energy": 1234., "forces": [[12.,34.,56.]] * n, "virial": [[42.] * 3] * 3}
    label_req = dict(req, dataset=str(dataset("changed-labels.jsonl", altered)))
    altered_out = work / "changed-labels.deviation.jsonl"
    run("Label independence", [engine, "model-deviation", request("labels", label_req), altered_out])
    assert lines(altered_out)[1:-1] == actual[1:-1]
    zero_req = dict(req, models=[members[0], dict(members[0], id="duplicate")], per_atom=False)
    zero_req.pop("force_window")
    zero_out = work / "zero.jsonl"
    run("Identical models and no selection window", [engine, "model-deviation", request("zero", zero_req), zero_out])
    zero = lines(zero_out)
    assert zero[0]["distinct_model_byte_hashes"] == 1 and zero[-1]["candidate_frames"] is None
    for entry in zero[1:-1]:
        s = entry["statistics"]
        assert entry["candidate"] is None and "per_atom" not in s["force_ev_per_angstrom"]
        assert s["energy_per_atom_std_ev"] == s["force_ev_per_angstrom"]["max"] == s["virial_per_atom_ev"]["max"] == 0
    rejections = []
    for name, change in [
        ("one-model", {"models": members[:1]}),
        ("duplicate-id", {"models": [members[0], members[0]]}),
        ("invalid-window", {"force_window": {"min_ev_per_angstrom": upper, "max_ev_per_angstrom": lower}}),
        ("unknown-field", {"relative": True}),
        ("cuda", {"models": [dict(m, options=dict(m["options"], device="cuda:0")) for m in members]}),
        ("truncated", {"dataset": str(dataset("truncated.jsonl", records[:-1]))}),
        ("type-map", {"dataset": str(dataset("wrong-types.jsonl", [dict(records[0], header=dict(header, type_map=list(reversed(header["type_map"]))))] + records[1:]))}),
    ]:
        bad_out = work / (name + ".output.jsonl")
        run("Reject " + name, [engine, "model-deviation", request(name, dict(req, **change)), bad_out], success=False)
        if bad_out.exists():
            assert all(r.get("kind") != "completed" for r in lines(bad_out))
        rejections.append(name)
    old = output.read_bytes()
    run("Reject existing output", [engine, "model-deviation", work / "ensemble.request.json", output], success=False)
    assert output.read_bytes() == old
    rejections.append("existing-output-preserved")
    # A reviewed candidate can enter the existing MD workflow by its original index.
    selected = candidates[0]
    md = {"options": members[0]["options"], "dataset_frame": {"dataset": str(data), "index": selected},
          "masses": [15.999 if s == "O" else 1.008 for s in frames[selected]["species"]],
          "velocities": [[0.,0.,0.]] * len(frames[selected]["species"]),
          "timestep_fs": 0.001, "steps": 2, "sample_every": 1}
    md_out = work / "candidate-md.jsonl"
    run("Candidate two-step CPU NVE", [engine, "run", request("candidate-md", md), md_out])
    assert lines(md_out)[-1]["kind"] == "completed"
    receipt = {"schema": "omnimd-model-deviation-receipt-v1", "status": "passed", "device": "cpu",
        "scope": "Original software fixtures only; native CPU ensemble. C API ensemble route not runtime-validated; no physical accuracy or calibrated uncertainty claim.",
        "models": 3, "frames": 12, "periodic_frames": 6, "nonperiodic_frames": 6, "atom_counts": [3,4],
        "population_oracle": "Python statistics.pstdev, independent vector norm and RMS aggregation",
        "maximum_absolute_statistic_error": max_error, "candidate_indices": candidates, "candidate_md_steps": 2,
        "label_independence": True, "identical_model_zero_deviation": True, "rejections_passed": rejections,
        "dataset_sha256": digest(data), "engine_sha256": digest(engine), "evaluator_sha256": digest(evaluator),
        "script_sha256": digest(__file__), "commands": commands}
    write(work / "report.json", receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k != "commands"}, indent=2))


if __name__ == "__main__":
    main()
