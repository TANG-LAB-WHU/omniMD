"""Continue the original model-deviation replay into selection, assessment and CPU fitting.
Requires validate_model_deviation.py outputs. No downloads, GPU use or overwrites.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines()]


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", required=True, type=Path)
    p.add_argument("--deviation-work", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path)
    a = p.parse_args()
    engine, previous, work = a.engine.resolve(), a.deviation_work.resolve(), a.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    data = previous / "dataset.jsonl"
    source = records(data)
    deviations = records(previous / "deviations.jsonl")
    assert deviations[-1]["kind"] == "completed" and deviations[-1]["dataset_sha256"] == digest(data)
    chosen = [r["id"] for r in deviations[1:-1] if r["candidate"]]
    commands = []

    def run(label, command, success=True):
        r = subprocess.run(list(map(str, command)), capture_output=True, encoding="utf-8", timeout=120)
        commands.append({"label": label, "exit_code": r.returncode})
        (work / f"step-{len(commands):02d}.stdout.txt").write_text(r.stdout, encoding="utf-8")
        (work / f"step-{len(commands):02d}.stderr.txt").write_text(r.stderr, encoding="utf-8")
        print(f"{len(commands)}. {label}: {r.returncode}", flush=True)
        assert (r.returncode == 0) == success, r.stderr
        return r

    def request(name, value):
        path = work / (name + ".json")
        write(path, value)
        return path

    def select(name, input_path, ids):
        req = request(name, {"schema": "omnimd-select-dataset-v1", "dataset": str(input_path),
                             "expected_source_sha256": digest(input_path), "ids": ids})
        bundle = work / name
        run(name, [engine, "select-dataset", req, bundle])
        receipt = read(bundle / "receipt.json")
        assert receipt["status"] == "completed"
        assert receipt["output_dataset_sha256"] == digest(bundle / "dataset.jsonl")
        assert receipt["request_sha256"] == digest(bundle / "request.json") == digest(req)
        assert receipt["input_dataset_sha256"] == digest(input_path)
        return bundle, receipt

    bundle, receipt = select("candidates", data, list(reversed(chosen)))
    selected = records(bundle / "dataset.jsonl")
    original_by_id = {r["data"]["id"]: r["data"] for r in source[1:-1]}
    assert [r["data"]["id"] for r in selected[1:-1]] == chosen
    for index, record in enumerate(selected[1:-1]):
        value = record["data"]
        original = dict(original_by_id[value["id"]], index=index)
        assert value == original
    run("Inspect selected data", [engine, "inspect-dataset", bundle / "dataset.jsonl", work / "inspection.json"])
    inspection = read(work / "inspection.json")
    assert inspection["frames"] == 6 and inspection["labels"]["energy_and_force_frames"] == 3
    options = {"native_model": str(previous / "model-0.json"), "device": "cpu"}
    assessment = request("assessment", {"options": options, "dataset": str(bundle / "dataset.jsonl")})
    run("Assess selected data", [engine, "batch-eval", assessment, work / "predictions.jsonl"])
    assert records(work / "predictions.jsonl")[-1]["kind"] == "completed"
    # Repeated selection keeps IDs but records indices relative to its immediate input.
    labeled = [r["data"]["id"] for r in selected[1:-1] if r["data"]["labels"]["energy"] is not None and r["data"]["labels"]["forces"] is not None]
    second, second_receipt = select("labeled", bundle / "dataset.jsonl", labeled)
    assert [m["source_index"] for m in second_receipt["mapping"]] == [0, 2, 5]
    config = {"split": {"mode": "explicit", "validation_ids": [labeled[-1]]}, "fit_bias": False,
              "energy_weight": 1., "force_weight": 1., "energy_scale_ev_per_atom": 0.1,
              "force_scale_ev_per_angstrom": 0.01, "ridge_inverse_ev2": 0., "rank_rtol": 1e-12}
    train = {"schema": "omnimd-native-readout-train-v1", "device": "cpu",
             "initial_model": str(previous / "model-1.json"), "dataset": str(second / "dataset.jsonl"),
             "output_directory": str(work / "trained"), "config": config}
    run("CPU train-native on labeled subset", [engine, "train-native", request("train", train)])
    trained = read(work / "trained/report.json")
    assert len(trained["training_frames"]) == 2 and len(trained["validation_frames"]) == 1
    failed_output = work / "missing-label-trained"
    bad_train = dict(train, dataset=str(bundle / "dataset.jsonl"), output_directory=str(failed_output))
    run("Refuse missing-label training", [engine, "train-native", request("missing-label", bad_train)], False)
    assert not failed_output.exists()
    # Wrong expected hash fails even though IDs all exist; no bundle is created.
    wrong_out = work / "wrong-hash"
    wrong = request("wrong-hash", {"schema": "omnimd-select-dataset-v1", "dataset": str(data),
                    "expected_source_sha256": "00" * 32, "ids": chosen})
    run("Refuse changed dataset identity", [engine, "select-dataset", wrong, wrong_out], False)
    assert not wrong_out.exists()
    before = digest(bundle / "dataset.jsonl")
    run("Refuse existing bundle", [engine, "select-dataset", work / "candidates.json", bundle], False)
    assert before == digest(bundle / "dataset.jsonl")
    result = {"schema": "omnimd-selection-replay-v1", "status": "passed", "device": "cpu",
              "scope": "Original synthetic software fixtures. No labels generated; CUDA-assisted trainer integration exercised on CPU only.",
              "source_frames": 12, "selected_frames": 6, "labeled_frames": 3, "training_frames": 2, "validation_frames": 1,
              "selected_ids": chosen, "selection_mapping": receipt["mapping"], "repeated_selection_mapping": second_receipt["mapping"],
              "scientific_content_exactly_preserved": True, "selection_order": "source order, despite reversed request IDs",
              "source_dataset_sha256": digest(data), "selected_dataset_sha256": digest(bundle / "dataset.jsonl"),
              "engine_sha256": digest(engine), "trainer_sha256": digest(engine.with_name("omnimd-native-train" + engine.suffix)),
              "script_sha256": digest(__file__), "training_data_rank": trained["data_rank"],
              "checks": commands}
    write(work / "report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
