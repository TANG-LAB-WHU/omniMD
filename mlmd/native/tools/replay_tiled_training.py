"""Original synthetic complete-process benchmark of bounded CUDA frame tiling.
Run only during an authorized GPU window. Python standard library; no downloads.
All datasets, weights, checkpoints and raw logs belong in an ignored output tree.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import subprocess
from replay_full_training import digest, perturb, run, write
from replay_indexed_training import launch
from replay_cuda_reverse import medium_model, values

def stripped(s):
    v=deepcopy(s)
    for key in ("schema","execution"): v.pop(key,None)
    for key in ("cuda_tiling","steps","fixed_geometry_cache"):v["config"].pop(key,None)
    return v

def prepare(root,out,evaluator,name):
    initial=medium_model(root) if name=="medium" else json.loads((root/"fixtures/full-tanh-v1.json").read_text())
    teacher=deepcopy(initial); parameters=perturb(teacher)
    write(out/"initial.json",initial);write(out/"teacher.json",teacher)
    frames=[]
    for k in range(128):
        if name=="tiny":
            f=json.loads((root/"fixtures/frame.jsonl").read_text())
            f["cell"]=None if k%2==0 else [[3.8,0.,0.],[.3,3.9,0.],[.1,.2,4.]]
            for atom,r in enumerate(f["positions"]):
                for a in range(3):r[a]+=.19*math.sin((k+1)*(atom*3+a+1)*.618)
            if k%3==0:f["positions"].pop();f["species"].pop()
        else:
            n=32 if k%3==0 else 64
            f={"positions":[[(i//2)*8.+(i%2)*(1.1+.04*math.sin(k+i)),(i%2)*.2,.1] for i in range(n)],
               "species":["O" if i%2==0 else "H" for i in range(n)],
               "cell":None if k%2==0 else [[264.,0.,0.],[.3,8.,0.],[.1,.2,8.]],"units":"EvAngstrom"}
        frames.append(f)
    labels=[json.loads(line) for line in run([evaluator,out/"teacher.json"],input="".join(json.dumps(f)+"\n" for f in frames)).splitlines()]
    header={"schema":"omnimd-dataset-jsonl-v1","type_map":initial["type_map"],"units":"EvAngstrom","virial_convention":"MinusDEpsilonRow",
            "source":"original synthetic tiled CUDA teacher; no physical accuracy claim","selected_sets":[]}
    records=[{"kind":"header","header":header}]
    records += [{"kind":"frame","data":{"index":i,"id":f"{name}/{i}","frame":f,"labels":{key:label[key] for key in ("energy","forces","virial")}}} for i,(f,label) in enumerate(zip(frames,labels))]
    records += [{"kind":"completed","frames":128,"source_sha256":{"teacher.json":digest(out/"teacher.json"),"replay_tiled_training.py":digest(__file__)}}]
    (out/"dataset.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in records),encoding="utf-8")
    return parameters,frames

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-exe",required=True,type=Path);p.add_argument("--train-exe",required=True,type=Path)
    p.add_argument("--reference-exe",required=True,type=Path,help="Retained pre-tile trainer; never rebuilt by this script")
    p.add_argument("--output-dir",required=True,type=Path);p.add_argument("--steps",type=int,default=128)
    a=p.parse_args()
    if a.steps<32 or a.steps%4:raise ValueError("steps must be >=32 and divisible by four")
    trainer=a.train_exe.resolve(); evaluator=a.eval_exe.resolve();reference=a.reference_exe.resolve()
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];cases=[]
    for name in ("tiny","medium"):
        folder=out/name;folder.mkdir()
        parameters,frames=prepare(root,folder,evaluator,name)
        cfg={"split":{"mode":"explicit","validation_ids":[f"{name}/{i}" for i in range(120,128)]},
             "steps":a.steps,"minibatch":{"batch_size":32,"seed":20261016,"order":"shuffle_each_epoch"},
             "cuda_gradient":"reverse","fixed_geometry_cache":True,"energy_weight":1.,"force_weight":1.,
             "virial_loss":{"weight":.7,"scale_ev_per_atom":.2},"energy_scale_ev_per_atom":.1,"force_scale_ev_per_angstrom":.1,
             "learning_rate":.002,"learning_rate_schedule":{"kind":"exponential","decay_steps":10,"decay_rate":.8},
             "beta1":.9,"beta2":.999,"epsilon":1e-8,"gradient_clip_norm":10.,"checkpoint_every":a.steps//2}
        def execute(label,config,exe=trainer,checkpoint=None,storage="indexed"):
            req={"schema":"omnimd-native-full-train-v1","device":"cuda:0","dataset_access":storage,
                 "dataset":"dataset.jsonl","output_directory":label,"config":config}
            req["checkpoint" if checkpoint else "initial_model"]=checkpoint or "initial.json"
            path=folder/(label+".request.json");write(path,req)
            process=launch(exe,path)
            report=json.loads((folder/label/"report.json").read_text())
            state=json.loads((folder/label/"checkpoint.json").read_text())
            assert report["native_v1_reload_max_abs_energy_force_virial"]==[0.0,0.0,0.0]
            print(json.dumps({"case":name,"run":label,"process":process,"gradient_ms":report["timing"]["gradient_ms"]}),flush=True)
            return state,report,process
        old,old_report,old_process=execute("historical",deepcopy(cfg),reference)
        states={};reports={};rows=[]
        for repeat in range(3):
            for mode in (("framewise","tiled") if repeat%2==0 else ("tiled","framewise")):
                c=deepcopy(cfg)
                if mode=="tiled":c["cuda_tiling"]={"max_frames":16,"max_atoms":512}
                label=f"{mode}-{repeat}";s,r,process=execute(label,c)
                assert stripped(s)==stripped(old),(name,mode,"historical numeric state changed")
                if mode in states:assert s==states[mode],(name,mode,"nonrepeatable checkpoint")
                states[mode]=s;reports[mode]=r
                prof=r["cuda_gradient_profile"];assert prof["frames"]==a.steps//4*120
                if mode=="tiled":
                    assert prof["max_frames_per_launch"]>1 and prof["tiles"]<prof["frames"]
                    assert prof["kernel_launches"]==prof["tiles"]*4
                    assert prof["synchronizations"]==prof["tiles"]
                    assert prof["peak_device_buffer_bytes"]<=64*1024*1024
                    assert prof["peak_host_tile_payload_bytes"]<=32*1024*1024
                    assert r["dataset_access"]["frames_decoded_per_read"]==1
                else:
                    assert prof["tiles"]==prof["frames"] and prof["max_frames_per_launch"]==1
                    assert r["execution"]==old_report["execution"],"legacy PTX/device identity changed"
                rows.append({"mode":mode,"repeat":repeat,"process":process,"timing":r["timing"],"gradient_profile":prof})
        c=deepcopy(states["tiled"]["config"]);c["steps"]=a.steps//2;c["fixed_geometry_cache"]=False
        resumed,rr,rp=execute("resumed",c,checkpoint=f"tiled-0/checkpoint-step-{a.steps//2}.json",storage="memory")
        assert stripped(resumed)==stripped(states["tiled"])
        assert resumed["execution"]==states["tiled"]["execution"]
        for key in ("final_training","final_validation"):assert rr[key]==reports["tiled"][key]
        rejected=[]
        for invalid in ("cpu","hyperdual","zero-frames","small-atoms","changed-caps","framewise-resume"):
            req=json.loads((folder/"tiled-0.request.json").read_text())
            req["output_directory"]="reject-"+invalid
            if invalid=="cpu":req["device"]="cpu"
            elif invalid=="hyperdual":req["config"]["cuda_gradient"]="hyperdual"
            elif invalid=="zero-frames":req["config"]["cuda_tiling"]["max_frames"]=0
            elif invalid=="small-atoms":req["config"]["cuda_tiling"]["max_atoms"]=1
            else:
                req.pop("initial_model");req["checkpoint"]=f"tiled-0/checkpoint-step-{a.steps//2}.json"
                if invalid=="changed-caps":req["config"]["cuda_tiling"]["max_frames"]=15
                else:req["config"].pop("cuda_tiling")
            path=folder/(invalid+".request.json");write(path,req)
            proc=subprocess.run([str(trainer),str(path)],capture_output=True,text=True)
            assert proc.returncode and not (folder/req["output_directory"]).exists(),invalid
            rejected.append({"case":invalid,"message":proc.stderr.strip()})
        medians={}
        for mode in ("framewise","tiled"):
            chosen=[r for r in rows if r["mode"]==mode]
            medians[mode]={"whole_process_ms":statistics.median(r["process"]["process_ms"] for r in chosen),
                           "gradient_ms":statistics.median(r["timing"]["gradient_ms"] for r in chosen),
                           "max_process_peak_working_set_bytes":max(r["process"]["peak_working_set_bytes"] or 0 for r in chosen)}
            medians[mode]["training_frames_per_whole_process_second"]=(a.steps//4*120)*1000/medians[mode]["whole_process_ms"]
        case={"case":name,"parameters":parameters,"dataset_frames":128,"training_frames":120,"validation_frames":8,
              "atoms_min":min(len(f["positions"]) for f in frames),"atoms_max":max(len(f["positions"]) for f in frames),
              "steps":a.steps,"processed_frames":a.steps//4*120,"medians":medians,
              "whole_process_speed_ratio":medians["framewise"]["whole_process_ms"]/medians["tiled"]["whole_process_ms"],
              "gradient_speed_ratio":medians["framewise"]["gradient_ms"]/medians["tiled"]["gradient_ms"],
              "runs":rows,"historical":{"process":old_process,"execution":old_report["execution"],"state_exact":True},
              "tile_resources":reports["tiled"]["cuda_tiling"],"tile_execution":reports["tiled"]["execution"],
              "geometry_cache":reports["tiled"]["fixed_geometry_cache"],"indexed_access":reports["tiled"]["dataset_access"],
              "resume":{"from_step":a.steps//2,"cache_off_memory_storage_exact":True,"process":rp},
              "repeat_and_cross_executor_state_exact":True,"rejected_before_output":rejected,
              "final_training":reports["tiled"]["final_training"],"final_validation":reports["tiled"]["final_validation"],
              "dataset_sha256":digest(folder/"dataset.jsonl"),"model_sha256":digest(folder/"tiled-0/model.json")}
        cases.append(case);write(out/"partial.json",cases)
    receipt={"scope":"bounded synthetic FP64 E/F/V multi-frame CUDA reverse training; no physical accuracy or production scaling claim",
             "process_measurement":"Complete subprocess wall time and Windows PeakWorkingSetSize, including startup, validation, assessment, checkpoint/report writes. Three alternating-order repeats; no performance threshold.",
             "budget_scope":"64 MiB explicitly allocated CUDA buffers and 32 MiB packed tile inputs/downloads/two gradient accumulators. Excludes CUDA context/module/local memory, allocator, CPU inference, one decoded frame, geometry cache/index, model/optimizer/report/checkpoint. PeakWorkingSetSize is measured, not a hard RSS cap.",
             "cases":cases,"sha256":{"script":digest(__file__),"trainer":digest(trainer),"evaluator":digest(evaluator),"historical_trainer":digest(reference)}}
    write(out/"acceptance.json",receipt)
    print(json.dumps({"complete":True,"cases":[{k:c[k] for k in ("case","whole_process_speed_ratio","gradient_speed_ratio")} for c in cases]}),flush=True)
if __name__=="__main__":main()
