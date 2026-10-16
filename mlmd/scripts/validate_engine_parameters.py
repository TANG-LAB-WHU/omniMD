"""CPU-only engine parameter acceptance against existing C API reference fixtures.
No downloads/reference rebuilding; use a compatible official DeepMD environment.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def npy(path, shape, values):
    header = repr({"descr": "<f8", "fortran_order": False, "shape": tuple(shape)})
    while (10 + len(header) + 1) % 64:
        header += " "
    payload = header.encode() + b"\n"
    with Path(path).open("xb") as stream:
        stream.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(payload)) + payload)
        stream.write(struct.pack("<" + "d" * len(values), *values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", required=True, type=Path)
    p.add_argument("--capi", required=True, type=Path)
    p.add_argument("--references", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path)
    a = p.parse_args()
    engine, capi, refs, work = a.engine.resolve(), a.capi.resolve(), a.references.resolve(), a.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    steps, comparisons = [], {}
    layout = {"fparam": [{"name": "official_fixture_frame_condition", "unit": "fixture_declared_numeric"}],
              "aparam": [{"name": "official_fixture_atom_condition", "unit": "fixture_declared_numeric"}]}

    def request(name, value):
        path = work / (name + ".request.json")
        write(path, value)
        return path

    def run(label, command, ok=True):
        r = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=180)
        steps.append({"label": label, "exit_code": r.returncode})
        (work / f"{len(steps):02}.stdout.txt").write_text(r.stdout)
        (work / f"{len(steps):02}.stderr.txt").write_text(r.stderr)
        print(f"{len(steps)}. {label}: {r.returncode}", flush=True)
        assert (r.returncode == 0) == ok, r.stderr
        return r

    def compare(name, first, second, tol):
        assert len(first) == len(second)
        maximum = [0., 0., 0.]
        for one, two in zip(first, second):
            x = [[one["energy"]], sum(one["forces"], []), sum(one["virial"], [])]
            y = [[two["energy"]], sum(two["forces"], []), sum(two["virial"], [])]
            for kind, (left, right) in enumerate(zip(x, y)):
                assert len(left) == len(right)
                for u, v in zip(left, right):
                    maximum[kind] = max(maximum[kind], abs(u-v))
                    assert abs(u-v) <= tol, (name, kind, u, v)
        comparisons[name] = {"max_absolute_efv": maximum, "absolute_tolerance": tol}

    datasets, requests = {}, {}
    for label in ["tf.serial", "pt.explicit", "pt.default"]:
        q = read(refs / (label + ".request.json"))
        # Keep model bytes; only adjust path to the explicitly supplied reference directory.
        q["options"]["model"] = str(refs / "fixtures" / Path(q["options"]["model"]).name)
        requests[label] = q
        path = work / (label + ".dataset.jsonl")
        if label == "tf.serial":
            system = work / "npy-system"
            (system / "set.000").mkdir(parents=True)
            (system / "type_map.raw").write_text("O\n")
            n = len(q["frames"][0]["positions"])
            (system / "type.raw").write_text(" ".join(["0"] * n) + "\n")
            nf = len(q["frames"])
            npy(system / "set.000/coord.npy", [nf, n*3], [x for f in q["frames"] for row in f["positions"] for x in row])
            npy(system / "set.000/box.npy", [nf,9], [x for f in q["frames"] for row in f["cell"] for x in row])
            npy(system / "set.000/fparam.npy", [nf,1], [x for v in q["parameters"] for x in v["fparam"]])
            npy(system / "set.000/aparam.npy", [nf,n], [x for v in q["parameters"] for row in v["aparam"] for x in row])
            imp = request("import", {"system": str(system), "sets": ["set.000"], "units": "EvAngstrom", "pbc": "Periodic", "parameter_layout": layout})
            run("Import explicit TF parameter NPY", [engine, "import-deepmd", imp, path])
        else:
            header = {"schema": "omnimd-dataset-jsonl-v2", "type_map": q["options"]["type_map"], "units": "EvAngstrom",
                      "virial_convention": "MinusDEpsilonRow", "source": "official existing parameter fixture", "selected_sets": [], "parameter_layout": layout}
            rows = [{"kind": "header", "header": header}]
            for i, (f, v) in enumerate(zip(q["frames"], q["parameters"])):
                rows.append({"kind": "frame", "data": {"index": i, "id": f"{label}/{i}", "frame": f,
                    "parameters": dict(v, frame_parameter_policy=q["frame_parameter_policy"]), "labels": {"energy": None, "forces": None, "virial": None}}})
            rows.append({"kind": "completed", "frames": len(q["frames"]), "source_sha256": {"official_request": sha(refs / (label + ".request.json"))}})
            with path.open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
        datasets[label] = path
        output = work / (label + ".predictions.jsonl")
        run("Engine " + label, [engine, "batch-eval", request(label, {"options": q["options"], "dataset": str(path)}), output])
        result = records(output)
        assert result[-1]["kind"] == "completed" and result[-1]["frames"] == len(q["frames"])
        values = [r["evaluation"] for r in result[1:-1]]
        tol = 1e-8 if label.startswith("tf") else 1e-7
        compare(label + "_saved_capi", values, read(refs / (label + ".result.json"))["evaluations"], tol)
        reference = {"tf.serial": "tf.python", "pt.explicit": "pt.python-explicit", "pt.default": "pt.python-default"}[label]
        compare(label + "_saved_python", values, read(refs / (reference + ".result.json"))["evaluations"], tol)
    tf = requests["tf.serial"]
    tf_predictions = records(work / "tf.serial.predictions.jsonl")
    response = abs(tf_predictions[2]["evaluation"]["energy"] - tf_predictions[1]["evaluation"]["energy"])
    assert response > 1e-4
    # Same-species atoms with distinct aparam values: permute coordinates and rows together.
    original = records(datasets["tf.serial"])
    permuted = deepcopy(original)
    permutation = [2, 0, 5, 1, 4, 3]
    for row in permuted[1:-1]:
        d = row["data"]
        for key in ["positions", "species"]:
            d["frame"][key] = [d["frame"][key][i] for i in permutation]
        d["parameters"]["aparam"] = [d["parameters"]["aparam"][i] for i in permutation]
    perm_path = work / "permuted.jsonl"
    with perm_path.open("x") as stream:
        for row in permuted:
            stream.write(json.dumps(row) + "\n")
    perm_out = work / "permuted.predictions.jsonl"
    run("Same-species coordinate and parameter permutation", [engine, "batch-eval", request("permuted", {"options": tf["options"], "dataset": str(perm_path)}), perm_out])
    returned = [r["evaluation"] for r in records(perm_out)[1:-1]]
    for value in returned:
        old = value["forces"]
        value["forces"] = [old[permutation.index(i)] for i in range(len(permutation))]
    compare("permutation", returned, [r["evaluation"] for r in tf_predictions[1:-1]], 1e-8)
    md_cases = []
    for label, index in [("tf.serial", 1), ("pt.default", 0)]:
        q = requests[label]
        n = len(q["frames"][index]["positions"])
        md = {"options": q["options"], "dataset_frame": {"dataset": str(datasets[label]), "index": index},
              "masses": [15.999]*n, "velocities": [[0.,0.,0.]]*n,
              "timestep_fs": 0.001, "steps": 4, "sample_every": 1, "checkpoint_every": 2}
        out = work / (label + ".md.jsonl")
        run(label + " fixed-parameter NVE", [engine, "run", request(label + "-md", md), out])
        full = records(out)
        assert full[-1]["kind"] == "completed" and full[0]["schema"] == "omnimd-mlmd-jsonl-v3"
        snaps = [r for r in full if r["kind"] == "frame"]
        initial = read(refs / (label + ".result.json"))["evaluations"][index]
        compare(label + "_md_initial", [snaps[0]["evaluation"]], [initial], 1e-8 if label.startswith("tf") else 1e-7)
        direct_md = deepcopy(md)
        direct_md.pop("dataset_frame")
        direct_md["frame"] = q["frames"][index]
        direct_md["fixed_parameters"] = deepcopy(full[0]["fixed_parameters"])
        direct_out_md = work / (label + ".direct-md.jsonl")
        run(label + " direct fixed-parameter request", [engine, "run", request(label + "-direct-md", direct_md), direct_out_md])
        assert [r for r in records(direct_out_md) if r["kind"] == "frame"] == snaps
        checkpoint = Path(str(out) + ".step-2.checkpoint.json")
        saved = read(checkpoint)
        assert saved["schema_version"] == 3 and saved["fixed_parameters"] == full[0]["fixed_parameters"]
        resume = {"options": q["options"], "restart": str(checkpoint), "timestep_fs": 0.001, "steps": 2, "sample_every": 1}
        resume_out = work / (label + ".resumed.jsonl")
        run(label + " fixed-context restart", [engine, "run", request(label + "-resume", resume), resume_out])
        resumed = [r for r in records(resume_out) if r["kind"] == "frame"]
        continuation = [r for r in snaps if r["step"] >= 2]
        restart_exact = resumed == continuation
        if label.startswith("tf"):
            assert restart_exact
        compare(label + "_restart_efv", [r["evaluation"] for r in resumed], [r["evaluation"] for r in continuation], 1e-12)
        state_error = {"positions_angstrom": 0., "velocities_angstrom_per_fs": 0.}
        for actual, expected in zip(resumed, continuation):
            assert actual["step"] == expected["step"]
            for key, x, y, tolerance in [("positions_angstrom", actual["frame"]["positions"], expected["frame"]["positions"], 1e-14),
                                         ("velocities_angstrom_per_fs", actual["velocities"], expected["velocities"], 1e-20)]:
                error = max(abs(u-v) for row, refrow in zip(x,y) for u,v in zip(row,refrow))
                state_error[key] = max(state_error[key], error)
                assert error <= tolerance
        cp_resume = read(Path(str(resume_out) + ".checkpoint.json"))
        cp_full = read(Path(str(out) + ".checkpoint.json"))
        checkpoint_exact = cp_resume == cp_full
        # Fixed context, step, cell, masses and model identity must remain exact.
        cp_resume["velocities"] = cp_full["velocities"]
        cp_resume["frame"]["positions"] = cp_full["frame"]["positions"]
        assert cp_resume == cp_full
        if label.startswith("tf"):
            assert checkpoint_exact
        # Independently recompute every moving frame through C's standalone API.
        values = q["parameters"][index]
        direct = {"options": q["options"], "frames": [r["frame"] for r in snaps], "parameters": [values]*len(snaps),
                  "frame_parameter_policy": q["frame_parameter_policy"], "batch_size": 1}
        direct_out = work / (label + ".md-capi.json")
        run(label + " all moving frames via standalone C API", [capi, request(label + "-md-capi", direct), direct_out])
        compare(label + "_md_all_frames", [r["evaluation"] for r in snaps], read(direct_out)["evaluations"], 1e-8 if label.startswith("tf") else 1e-7)
        bad = deepcopy(resume)
        bad["fixed_parameters"] = deepcopy(saved["fixed_parameters"])
        bad["fixed_parameters"]["values"]["aparam"][0][0] += 0.1
        bad_out = work / (label + ".bad-restart.jsonl")
        run(label + " reject parameter-changing restart", [engine, "run", request(label + "-bad-restart", bad), bad_out], False)
        assert not bad_out.exists()
        md_cases.append({"label": label, "steps": 4, "restart_step": 2, "same_environment_restart_exact": restart_exact,
                         "final_checkpoint_exact": checkpoint_exact, "restart_state_max_absolute_error": state_error,
                         "policy": q["frame_parameter_policy"], "same_species_distinct_aparam": label.startswith("tf")})
    # Isolate PT first-call roundoff using the identical saved geometry, with no engine integrator.
    checkpoint = read(work / "pt.default.md.jsonl.step-2.checkpoint.json")
    pt = requests["pt.default"]
    repeat = {"options": pt["options"], "frames": [checkpoint["frame"]]*4, "parameters": [pt["parameters"][0]]*4,
              "frame_parameter_policy": "model_default", "batch_size": 1}
    repeated_out = work / "pt-repeated-identical-frame.json"
    run("PT standalone identical-frame repeat diagnostic", [capi, request("pt-repeat", repeat), repeated_out])
    repeated = read(repeated_out)["evaluations"]
    compare("pt_first_vs_subsequent_identical_frame", [repeated[0]]*3, repeated[1:], 1e-12)
    # TF defaults must remain rejected; engine does not fall back to explicit zeros.
    bad_records = deepcopy(original)
    for row in bad_records[1:-1]:
        row["data"]["parameters"]["fparam"] = []
        row["data"]["parameters"]["frame_parameter_policy"] = "model_default"
    bad_path = work / "tf-default-invalid.jsonl"
    with bad_path.open("x") as stream:
        for row in bad_records:
            stream.write(json.dumps(row) + "\n")
    bad_out = work / "tf-default-invalid.predictions.jsonl"
    run("Reject unsupported TF defaults", [engine, "batch-eval", request("tf-default-invalid", {"options": tf["options"], "dataset": str(bad_path)}), bad_out], False)
    assert not bad_out.exists() or all(r["kind"] != "completed" for r in records(bad_out))
    native_options = {"native_model": str(Path(__file__).resolve().parents[1] / "native/fixtures/synthetic-v1.json"), "device": "cpu"}
    for name, options in [("native-parameters", native_options), ("cuda-parameters", dict(tf["options"], device="cuda:0"))]:
        output = work / (name + ".jsonl")
        run("Reject " + name + " before backend load", [engine, "batch-eval", request(name, {"options": options, "dataset": str(datasets["tf.serial"])}), output], False)
        assert not output.exists()
    dev_out = work / "unsupported-deviation.jsonl"
    dev_req = {"schema": "omnimd-model-deviation-v1", "models": [{"id": str(i), "options": tf["options"]} for i in range(2)], "dataset": str(datasets["tf.serial"])}
    run("Reject parameterized ensemble rather than discard context", [engine, "model-deviation", request("unsupported-deviation", dev_req), dev_out], False)
    assert not dev_out.exists()
    report = {"schema": "omnimd-engine-parameters-receipt-v1", "status": "passed", "device": "cpu",
              "scope": "Official DeepMD 3.1.3 saved TF/PT fixtures; v2 NPY/JSONL import, serial engine batch and fixed-parameter NVE/restart. No GPU/performance/chemical-accuracy claim.",
              "engine_sha256": sha(engine), "capi_cli_sha256": sha(capi), "script_sha256": sha(__file__),
              "comparisons": comparisons, "parameter_energy_response_ev": response, "md_cases": md_cases,
              "reference_hashes": {name: sha(refs / name) for name in ["tf.serial.result.json", "tf.python.result.json", "pt.default.result.json", "pt.python-default.result.json", "fixtures/fparam_aparam-tmap-O.pb", "fixtures/fparam_aparam_default.pth", "fixtures/manifest.json"]},
              "pt_restart_limitation": "PT runtime first call versus subsequent identical-frame calls can differ at floating-point roundoff. Parameter/geometry restore is exact; PT MD restart is numerical, not bitwise, parity. TF tested restart is bitwise exact.",
              "provenance": "TF fixture has only official declared tmap metadata supplied by C, weights unchanged; original C provenance manifest retained. PT model unmodified.", "steps": steps}
    write(work / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
