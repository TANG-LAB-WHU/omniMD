#!/usr/bin/env python3
"""Real CPU C-API + omniMD integration check using one official public model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ[key] = "-1"
os.environ.setdefault("DP_INTRA_OP_PARALLELISM_THREADS", "1")
os.environ.setdefault("DP_INTER_OP_PARALLELISM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

SOURCE = "https://raw.githubusercontent.com/deepmodeling/deepmd-kit/v3.1.3/source/tests/infer/deeppot.pbtxt"
SOURCE_SHA256 = "89e6aaa544dafc4620ecf49340ac2cd9e74ca0cfc2aae2c7bb060028550283a3"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--eval-bin", type=Path, required=True)
    parser.add_argument("--md-bin", type=Path, required=True)
    parser.add_argument("--pbtxt", type=Path)
    parser.add_argument("--timestep-fs", type=float, default=0.01)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import deepmd
    import tensorflow as tf
    from deepmd.infer import DeepPot
    from google.protobuf import text_format
    from tensorflow.core.framework.graph_pb2 import GraphDef
    assert deepmd.__version__ == "3.1.3", deepmd.__version__
    raw = args.pbtxt.read_bytes() if args.pbtxt else urllib.request.urlopen(SOURCE).read()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256, "public fixture checksum changed"
    graph = GraphDef()
    text_format.Parse(raw.decode(), graph)
    floating_constants = {
        tf.dtypes.as_dtype(n.attr["value"].tensor.dtype).name
        for n in graph.node if n.op == "Const"
        and n.attr["value"].tensor.dtype in (tf.float16.as_datatype_enum,
             tf.float32.as_datatype_enum, tf.float64.as_datatype_enum)
    }
    assert floating_constants == {"float64"}, floating_constants
    coord_node = next(n for n in graph.node if n.name == "t_coord")
    assert coord_node.attr["dtype"].type == tf.float64.as_datatype_enum
    model_path = args.work / "public-deeppot.pb"
    model_path.write_bytes(graph.SerializeToString())
    reference = DeepPot(str(model_path))
    assert reference.get_type_map() == ["O", "H"]
    options = {"library": str(args.library.resolve()), "model": str(model_path.resolve()),
               "device": "Cpu", "type_map": ["O", "H"], "units": "EvAngstrom"}
    positions = np.array([[1.,1.,1.],[1.9572,1.,1.],[0.761,1.927,1.],
                          [4.,4.,4.],[4.9572,4.,4.],[3.761,4.927,4.]],dtype=np.float64)
    species = ["O","H","H","O","H","H"]
    skew = np.array([[9.,0.,0.],[1.2,8.5,0.],[0.4,0.7,8.]],dtype=np.float64)
    def frame(x, symbols=species, cell=None):
        return {"positions": np.asarray(x).tolist(), "species": symbols,
                "cell": None if cell is None else np.asarray(cell).tolist(), "units":"EvAngstrom"}
    permutation = [2,0,1,5,3,4]
    frames = [frame(positions), frame(positions,cell=np.eye(3)*9),
              frame(positions,cell=skew),
              frame(positions[permutation],[species[i] for i in permutation],skew)]
    def evaluate(f):
        x=np.asarray(f["positions"],dtype=np.float64)
        cell=None if f["cell"] is None else np.asarray(f["cell"],dtype=np.float64).reshape(1,9)
        atom_types=[options["type_map"].index(s) for s in f["species"]]
        e,force,v=reference.eval(x.reshape(1,-1),cell,atom_types)
        assert e.dtype == force.dtype == v.dtype == np.float64
        return {"energy":float(e.reshape(-1)[0]), "forces":force.reshape(-1,3),
                "virial":v.reshape(3,3)}
    def run(name, request, binary, extension="json", success=True):
        request_path=args.work/(name+".request.json")
        result=args.work/(name+"."+extension)
        request_path.write_text(json.dumps(request),encoding="utf8")
        process=subprocess.run([str(binary),str(request_path),str(result)],
            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (args.work/(name+".log")).write_text(process.stdout,encoding="utf8")
        if success:
            if process.returncode != 0:
                raise RuntimeError(process.stdout)
            assert result.exists()
        else:
            assert process.returncode != 0 and not result.exists(), process.stdout
        return result
    path=run("eval",{"options":options,"frames":frames},args.eval_bin)
    output=json.loads(path.read_text())
    max_error={key:0.0 for key in ("energy","forces","virial")}
    for f,result in zip(frames,output["evaluations"],strict=True):
        expected=evaluate(f)
        for key in max_error:
            err=float(np.max(np.abs(np.asarray(result[key])-expected[key])))
            max_error[key]=max(max_error[key],err)
            assert err<1e-8, (key,err)
    # Independent finite differences of the model energy, including off-diagonal strain.
    h=1e-5
    p=positions.copy(); p[0,0]+=h
    m=positions.copy(); m[0,0]-=h
    force_fd=-(evaluate(frame(p,cell=skew))["energy"]-evaluate(frame(m,cell=skew))["energy"])/(2*h)
    deform=np.eye(3); deform[0,1]+=h
    eminus=np.eye(3); eminus[0,1]-=h
    virial_fd=-(evaluate(frame(positions@deform,cell=skew@deform))["energy"]
                -evaluate(frame(positions@eminus,cell=skew@eminus))["energy"])/(2*h)
    assert abs(force_fd-output["evaluations"][2]["forces"][0][0])<5e-5
    assert abs(virial_fd-output["evaluations"][2]["virial"][0][1])<5e-5

    for name,change in (("bad-map",{"type_map":["H","O"]}),
                        ("missing-model",{"model":str(args.work/"missing.pb")})):
        run(name,{"options":dict(options,**change),"frames":[frames[0]]},args.eval_bin,success=False)
    bad_model=args.work/"invalid.pb"; bad_model.write_bytes(b"not a protobuf model")
    run("invalid-model",{"options":dict(options,model=str(bad_model)),"frames":[frames[0]]},
        args.eval_bin,success=False)

    masses=np.array([15.999,1.008,1.008,15.999,1.008,1.008])
    velocities=np.array([[0.0001,0.,0.],[-0.0002,0.0001,0.],[0.0001,-0.0001,0.],
                         [-0.0001,0.,0.],[0.0002,-0.0001,0.],[-0.0001,0.0001,0.]])
    dt=args.timestep_fs; steps=100
    request={"options":options,"frame":frames[2],"masses":masses.tolist(),
        "velocities":velocities.tolist(),"timestep_fs":dt,"steps":steps,"sample_every":10}
    trajectory=run("nve",request,args.md_bin,"jsonl")
    records=[json.loads(line) for line in trajectory.read_text().splitlines()]
    assert records[-1]["kind"]=="completed"
    # Check the drift after writing the complete comparison report.
    final=next(r for r in reversed(records) if r["kind"]=="frame")
    # Separate Python implementation of velocity-Verlet using official DeepMD forces.
    # This is the exact legacy omniMD unit factor, not a new physical-unit convention.
    ev_to_internal=1.60217653e-19*1e-10/1.66053906660e-27
    x=positions.copy(); v=velocities.copy()
    f=evaluate(frame(x,cell=skew))["forces"]
    for _ in range(steps):
        v+=0.5*dt*f*ev_to_internal/masses[:,None]
        x+=dt*v
        f=evaluate(frame(x,cell=skew))["forces"]
        v+=0.5*dt*f*ev_to_internal/masses[:,None]
    position_error=float(np.max(np.abs(x-np.asarray(final["frame"]["positions"]))))
    velocity_error=float(np.max(np.abs(v-np.asarray(final["velocities"]))))
    assert position_error<1e-9 and velocity_error<1e-9
    expected_final=evaluate(final["frame"])
    for key in max_error:
        assert np.max(np.abs(np.asarray(final["evaluation"][key])-expected_final[key]))<1e-8
    report={"deepmd_version":deepmd.__version__,"tensorflow_version":tf.__version__,
        "fixture_url":SOURCE,"fixture_sha256":SOURCE_SHA256,
        "model_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "library_sha256":hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "model_float_constants":sorted(floating_constants),"coordinate_dtype":"float64",
        "evaluated_frames":len(frames),"max_absolute_error_vs_python":max_error,
        "force_fd_error":abs(force_fd-output["evaluations"][2]["forces"][0][0]),
        "offdiagonal_virial_fd_error":abs(virial_fd-output["evaluations"][2]["virial"][0][1]),
        "nve":dict(records[-1], timestep_fs=dt, duration_fs=dt*steps),"nve_final_position_error_angstrom":position_error,
        "nve_final_velocity_error_angstrom_per_fs":velocity_error,
        "boundary_errors_checked":["bad-map","missing-model","invalid-model"],
        "device":"CPU","private_inputs_used":False}
    (args.work/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,indent=2))
    assert records[-1]["max_absolute_energy_drift_ev"]<1e-3, records[-1]

if __name__=="__main__":
    main()
