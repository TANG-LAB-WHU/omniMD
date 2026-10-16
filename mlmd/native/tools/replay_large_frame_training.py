"""Original synthetic larger-frame CLI acceptance; Python stdlib and existing Rust binaries.
Run only during an authorized GPU window. Generated files must remain ignored.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
from replay_full_training import digest, perturb, run, write
from replay_indexed_training import launch
from replay_cuda_reverse import medium_model, values


def base_frame():
    cell = [[14.4, 0., 0.], [.4, 7.2, 0.], [.2, .3, 7.2]]
    f = dict(positions=[], species=[], cell=cell, units="EvAngstrom")
    for x in range(4):
        for y in range(2):
            for z in range(2):
                for t in range(2):
                    frac = [x/4, y/2, z/2]; k = len(f["positions"])
                    f["positions"].append([sum(frac[b]*cell[b][a] for b in range(3))
                        + t*[1.07, .21, .13][a] + .03*math.sin((k*3+a+1)*.61) for a in range(3)])
                    f["species"].append("O" if t == 0 else "H")
    return f


def replicate(f, factors):
    c = f["cell"]; result = dict(positions=[], species=[], cell=[[v*factors[a] for v in row] for a,row in enumerate(c)], units="EvAngstrom")
    for x in range(factors[0]):
        for y in range(factors[1]):
            for z in range(factors[2]):
                count = [x,y,z]
                for pos,typ in zip(f["positions"],f["species"]):
                    result["positions"].append([pos[a]+sum(count[b]*c[b][a] for b in range(3)) for a in range(3)])
                    result["species"].append(typ)
    return result


def checkpoint_error(a,b):
    av=values(a["model"])+a["first_moment"]+a["second_moment"]
    bv=values(b["model"])+b["first_moment"]+b["second_moment"]
    assert len(av)==len(bv)
    return max(abs(x-y) for x,y in zip(av,bv))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("eval-exe","train-exe","reference-exe","output-dir"):p.add_argument("--"+key,required=True,type=Path)
    a=p.parse_args(); evaluator=a.eval_exe.resolve();trainer=a.train_exe.resolve();reference=a.reference_exe.resolve()
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    m=json.loads((root/"fixtures/full-tanh-v1.json").read_text());teacher=deepcopy(m);assert perturb(teacher)==112
    write(out/"initial.json",m);write(out/"teacher.json",teacher);write(out/"medium.json",medium_model(root))
    base=base_frame();frames=[replicate(base,f) for f in ([2,2,1],[2,2,2],[1,1,1],[2,2,1],[2,2,2],[2,2,1],[2,2,1],[2,2,2])]
    # Perturb complete supercells to make the training replay heterogeneous, beyond pure repetition.
    for i,f in enumerate(frames):
        for j,pos in enumerate(f["positions"]):
            for axis in range(3):pos[axis]+=.015*math.sin((i+1)*(3*j+axis+1)*.41)
    def make_data(path,geometries):
        labels=[json.loads(line) for line in run([evaluator,out/"teacher.json"],input="".join(json.dumps(f)+"\n" for f in geometries)).splitlines()]
        records=[{"kind":"header","header":{"schema":"omnimd-dataset-jsonl-v1","type_map":m["type_map"],"units":"EvAngstrom",
            "virial_convention":"MinusDEpsilonRow","source":"original bounded larger-frame synthetic teacher","selected_sets":[]}}]
        records += [{"kind":"frame","data":{"index":i,"id":f"large/{i}","frame":f,"labels":{k:l[k] for k in ("energy","forces","virial")}}} for i,(f,l) in enumerate(zip(geometries,labels))]
        records += [{"kind":"completed","frames":len(geometries),"source_sha256":{"teacher.json":digest(out/"teacher.json"),"replay_large_frame_training.py":digest(__file__)}}]
        path.write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in records),encoding="utf-8")
    make_data(out/"dataset.jsonl",frames);make_data(out/"small.jsonl",[base,base,base])
    config={"split":{"mode":"explicit","validation_ids":["large/6","large/7"]},"steps":12,
        "minibatch":{"batch_size":4,"seed":20261016,"order":"shuffle_each_epoch"},
        "cuda_gradient":"reverse","cuda_tiling":{"max_frames":4,"max_atoms":512,"max_atoms_per_frame":256},
        "cuda_loss":"gpu","fixed_geometry_cache":True,"energy_weight":.7,"force_weight":.8,
        "virial_loss":{"weight":1.3,"scale_ev_per_atom":.4},"energy_scale_ev_per_atom":.3,"force_scale_ev_per_angstrom":.2,
        "learning_rate":.003,"learning_rate_schedule":{"kind":"exponential","decay_steps":3,"decay_rate":.8},
        "beta1":.9,"beta2":.999,"epsilon":1e-8,"gradient_clip_norm":.1,"checkpoint_every":6}
    def request(label,cfg):return {"schema":"omnimd-native-full-train-v1","device":"cuda:0","dataset_access":"indexed",
        "dataset":"dataset.jsonl","initial_model":"initial.json","output_directory":label,"config":deepcopy(cfg)}
    def execute(req,exe=trainer):
        path=out/(req["output_directory"]+".request.json");write(path,req);process=launch(exe,path)
        folder=out/req["output_directory"];state=json.loads((folder/"checkpoint.json").read_text());report=json.loads((folder/"report.json").read_text())
        assert report["native_v1_reload_max_abs_energy_force_virial"]==[0.,0.,0.]
        row={"label":req["output_directory"],"process":process,"timing":report["timing"],"profile":report["cuda_gradient_profile"],
             "execution":report["execution"],"resources":report["cuda_tiling"],"geometry_cache":report.get("fixed_geometry_cache"),"dataset_access":report.get("dataset_access")}
        print(json.dumps({"label":row["label"],"process":process,"gradient_ms":report["timing"]["gradient_ms"]}),flush=True)
        return state,report,row
    cpu=request("large-cpu-loss",config);cpu["config"]["cuda_loss"]="cpu"
    sc,rc,pc=execute(cpu);gpu=request("large-gpu-loss",config);sg,rg,pg=execute(gpu)
    assert sc["schema"]==sg["schema"]=="omnimd-native-full-checkpoint-v9"
    error=checkpoint_error(sc,sg);assert error<1e-10
    assert rg["execution"]["backend"]=="cuda_tiled_device_loss_f64_v2"
    for row in (pc,pg):
        assert row["profile"]["max_atoms_per_frame"]==256
        assert row["profile"]["peak_device_buffer_bytes"]<=64*1024*1024
        assert row["profile"]["peak_host_tile_payload_bytes"]<=32*1024*1024
        assert row["resources"]["estimated_scalar_terms_at_capacity"]<=50_000_000
    resume=request("resumed-v9",config);resume.pop("initial_model");resume["checkpoint"]="large-gpu-loss/checkpoint-step-6.json"
    resume["dataset_access"]="memory";resume["config"]["steps"]=6;resume["config"]["fixed_geometry_cache"]=False
    sr,rr,pr=execute(resume)
    def comparable(s):
        s=deepcopy(s)
        for k in ("steps","fixed_geometry_cache"):s["config"].pop(k,None)
        return s
    assert comparable(sr)==comparable(sg)
    assert rr["final_training"]==rg["final_training"] and rr["final_validation"]==rg["final_validation"]
    small=request("historical-small",config);small["dataset"]="small.jsonl";small["config"]["split"]["validation_ids"]=["large/2"]
    small["config"]["steps"]=2;small["config"]["checkpoint_every"]=1;small["config"]["cuda_tiling"].pop("max_atoms_per_frame")
    so,ro,po=execute(small,reference);small["output_directory"]="current-small";sn,rn,pn=execute(small)
    assert so==sn and ro["execution"]==rn["execution"] and sn["schema"]=="omnimd-native-full-checkpoint-v8"
    errors=[]
    for case in ("default-frame-cap","frame-cap-257","tile-cap-255","work-budget","cpu-device","changed-cap-resume","old-schema-resume","changed-ptx-resume"):
        q=request("reject-"+case,config)
        if case=="default-frame-cap":q["config"]["cuda_tiling"].pop("max_atoms_per_frame")
        elif case=="frame-cap-257":q["config"]["cuda_tiling"]["max_atoms_per_frame"]=257
        elif case=="tile-cap-255":q["config"]["cuda_tiling"]["max_atoms"]=255
        elif case=="work-budget":q["initial_model"]="medium.json"
        elif case=="cpu-device":q["device"]="cpu"
        else:
            q.pop("initial_model");checkpoint=json.loads((out/"large-gpu-loss/checkpoint-step-6.json").read_text())
            if case=="changed-cap-resume":q["config"]["cuda_tiling"]["max_atoms"]=511
            elif case=="old-schema-resume":checkpoint["schema"]="omnimd-native-full-checkpoint-v8"
            else:checkpoint["execution"]["program_sha256"]="0"*64
            name=case+".checkpoint.json";write(out/name,checkpoint);q["checkpoint"]=name
        path=out/(case+".request.json");write(path,q);proc=subprocess.run([str(trainer),str(path)],capture_output=True,text=True)
        assert proc.returncode and not (out/q["output_directory"]).exists(),case
        errors.append({"case":case,"message":proc.stderr.strip()})
    # Indexed parsing cap: well-shaped 257-atom frame, independently of label shape failures.
    records=[json.loads(s) for s in (out/"dataset.jsonl").read_text().splitlines()];r=records[2]["data"]
    r["frame"]["positions"].append([900.,0.,0.]);r["frame"]["species"].append("O");r["labels"]["forces"].append([0.,0.,0.])
    (out/"over257.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    q=request("reject-parser-257",config);q["dataset"]="over257.jsonl";path=out/"parser-257.request.json";write(path,q)
    proc=subprocess.run([str(trainer),str(path)],capture_output=True,text=True)
    assert proc.returncode and "array bound exceeded (maximum 256)" in proc.stderr and not (out/q["output_directory"]).exists()
    errors.append({"case":"indexed-parser-257","message":proc.stderr.strip()})
    result={"scope":"original synthetic bounded larger-frame acceptance; one observed process run per mode, no speedup claim",
        "frames":[len(f["positions"]) for f in frames],"train_frames":6,"validation_frames":2,"steps":12,"processed_frames":36,
        "runs":[pc,pg,pr,po,pn],"gpu_vs_cpu_loss_final_parameters_moments_max_abs":error,
        "v9_resume_cache_and_storage_switch_exact":True,"historical_default_v8_state_and_ptx_exact":True,"rejections":errors,
        "native_v1_reload_EFV_difference":[0.,0.,0.],"final_training":rg["final_training"],"final_validation":rg["final_validation"],
        "sha256":{"script":digest(__file__),"trainer":digest(trainer),"evaluator":digest(evaluator),"reference":digest(reference),"dataset":digest(out/"dataset.jsonl"),"exported_model":digest(out/"large-gpu-loss/model.json")}}
    write(out/"acceptance.json",result);print(json.dumps({"complete":True,"parameter_moment_error":error,"resume_exact":True,"rejections":len(errors)}),flush=True)


if __name__=="__main__":main()
