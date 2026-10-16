"""Original synthetic whole-process GPU prediction/seed comparison.
Uses only existing Rust binaries and Python stdlib. Run in an authorized GPU window.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import subprocess
from replay_full_training import digest,write
from replay_indexed_training import launch
from replay_tiled_training import prepare
from replay_cuda_reverse import values

def stripped(s):
    v=deepcopy(s)
    for key in ("schema","execution"):v.pop(key,None)
    for key in ("cuda_loss","steps","fixed_geometry_cache"):v["config"].pop(key,None)
    return v
def numeric_error(a,b):
    aa=values(a["model"])+a["first_moment"]+a["second_moment"]
    bb=values(b["model"])+b["first_moment"]+b["second_moment"]
    return max(abs(x-y) for x,y in zip(aa,bb))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-exe",type=Path,required=True);p.add_argument("--eval-exe",type=Path,required=True)
    p.add_argument("--reference-exe",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args();trainer=a.train_exe.resolve();evaluator=a.eval_exe.resolve();reference=a.reference_exe.resolve()
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];cases=[]
    for name in ("tiny","medium"):
        folder=out/name;folder.mkdir();parameters,frames=prepare(root,folder,evaluator,name)
        c={"split":{"mode":"explicit","validation_ids":[f"{name}/{i}" for i in range(120,128)]},
           "steps":128,"minibatch":{"batch_size":32,"seed":20261016,"order":"shuffle_each_epoch"},
           "cuda_gradient":"reverse","cuda_tiling":{"max_frames":16,"max_atoms":512},"fixed_geometry_cache":True,
           "energy_weight":1.,"force_weight":1.,"virial_loss":{"weight":.7,"scale_ev_per_atom":.2},
           "energy_scale_ev_per_atom":.1,"force_scale_ev_per_angstrom":.1,"learning_rate":.002,
           "learning_rate_schedule":{"kind":"exponential","decay_steps":10,"decay_rate":.8},
           "beta1":.9,"beta2":.999,"epsilon":1e-8,"gradient_clip_norm":10.,"checkpoint_every":64}
        def execute(label,config,exe=trainer,checkpoint=None,storage="indexed"):
            req={"schema":"omnimd-native-full-train-v1","device":"cuda:0","dataset_access":storage,"dataset":"dataset.jsonl","output_directory":label,"config":config}
            req["checkpoint" if checkpoint else "initial_model"]=checkpoint or "initial.json"
            path=folder/(label+".request.json");write(path,req);process=launch(exe,path)
            report=json.loads((folder/label/"report.json").read_text());state=json.loads((folder/label/"checkpoint.json").read_text())
            assert report["native_v1_reload_max_abs_energy_force_virial"]==[0.,0.,0.]
            print(json.dumps({"case":name,"run":label,"process":process,"gradient_ms":report["timing"]["gradient_ms"]}),flush=True)
            return state,report,process
        old,old_report,old_process=execute("historical-v7",deepcopy(c),reference)
        states={};reports={};rows=[]
        for repeat in range(3):
            for mode in (("cpu-loss","gpu-loss") if repeat%2==0 else ("gpu-loss","cpu-loss")):
                cfg=deepcopy(c)
                if mode=="gpu-loss":cfg["cuda_loss"]="gpu"
                s,r,proc=execute(f"{mode}-{repeat}",cfg)
                if mode=="cpu-loss":
                    assert stripped(s)==stripped(old),"historical v7 state changed"
                    assert r["execution"]==old_report["execution"],"historical v7 PTX changed"
                else:
                    assert numeric_error(s,old)<1e-10,"GPU prediction training diverges from reference"
                    assert s["schema"]=="omnimd-native-full-checkpoint-v8"
                    prof=r["cuda_gradient_profile"]
                    assert prof["device_prediction_frames"]==3840 and prof["prediction_and_seeds_ms"]==0.
                    assert prof["kernel_launches"]==10*prof["tiles"]
                    assert prof["synchronizations"]==prof["tiles"]
                    assert prof["max_frames_per_launch"]>1
                    assert r["cuda_tiling"]["estimated_scalar_terms_at_capacity"]<=50_000_000
                    assert prof["peak_device_buffer_bytes"]<=64*1024*1024
                    assert prof["peak_host_tile_payload_bytes"]<=32*1024*1024
                if mode in states:assert s==states[mode],f"{mode} is not deterministic"
                states[mode]=s;reports[mode]=r
                rows.append({"mode":mode,"repeat":repeat,"process":proc,"timing":r["timing"],"profile":r["cuda_gradient_profile"]})
        cfg=deepcopy(states["gpu-loss"]["config"]);cfg["steps"]=64;cfg["fixed_geometry_cache"]=False
        resumed,rr,rp=execute("resumed-v8",cfg,checkpoint="gpu-loss-0/checkpoint-step-64.json",storage="memory")
        assert stripped(resumed)==stripped(states["gpu-loss"]);assert resumed["execution"]==states["gpu-loss"]["execution"]
        assert rr["final_training"]==reports["gpu-loss"]["final_training"] and rr["final_validation"]==reports["gpu-loss"]["final_validation"]
        errors=[]
        for kind in ("cpu-device","missing-tiling","hyperdual","loss-mode-resume","v7-to-v8","small-capacity"):
            req=json.loads((folder/"gpu-loss-0.request.json").read_text());req["output_directory"]="reject-"+kind
            if kind=="cpu-device":req["device"]="cpu"
            elif kind=="missing-tiling":req["config"].pop("cuda_tiling")
            elif kind=="hyperdual":req["config"]["cuda_gradient"]="hyperdual"
            elif kind=="small-capacity":req["config"]["cuda_tiling"]["max_atoms"]=1
            else:
                req.pop("initial_model")
                req["checkpoint"]="gpu-loss-0/checkpoint-step-64.json" if kind=="loss-mode-resume" else "cpu-loss-0/checkpoint-step-64.json"
                if kind=="loss-mode-resume":req["config"].pop("cuda_loss")
            path=folder/(kind+".request.json");write(path,req)
            proc=subprocess.run([str(trainer),str(path)],capture_output=True,text=True)
            assert proc.returncode and not (folder/req["output_directory"]).exists(),kind
            errors.append({"case":kind,"message":proc.stderr.strip()})
        medians={}
        for mode in states:
            chosen=[r for r in rows if r["mode"]==mode]
            medians[mode]={"whole_process_ms":statistics.median(r["process"]["process_ms"] for r in chosen),
                "gradient_ms":statistics.median(r["timing"]["gradient_ms"] for r in chosen),
                "max_process_peak_working_set_bytes":max(r["process"]["peak_working_set_bytes"] or 0 for r in chosen)}
            medians[mode]["frames_per_whole_process_second"]=3840*1000/medians[mode]["whole_process_ms"]
        case={"case":name,"parameters":parameters,"frames":128,"training_frames":120,"validation_frames":8,"processed_frames":3840,"steps":128,
            "atoms_min":min(len(f["positions"]) for f in frames),"atoms_max":max(len(f["positions"]) for f in frames),
            "medians":medians,"whole_process_speed_ratio":medians["cpu-loss"]["whole_process_ms"]/medians["gpu-loss"]["whole_process_ms"],
            "gradient_speed_ratio":medians["cpu-loss"]["gradient_ms"]/medians["gpu-loss"]["gradient_ms"],"runs":rows,
            "historical":{"process":old_process,"execution":old_report["execution"],"current_cpu_loss_state_and_ptx_exact":True},
            "gpu_loss_execution":reports["gpu-loss"]["execution"],"gpu_loss_resources":reports["gpu-loss"]["cuda_tiling"],
            "gpu_loss_vs_historical_parameters_and_moments_max_abs":numeric_error(states["gpu-loss"],old),"same_backend_repeats_exact":True,
            "resume":{"step":64,"cache_off_memory_storage_state_and_metrics_exact":True,"process":rp},
            "rejections":errors,"native_v1_reload_EFV_difference":[0.,0.,0.],
            "final_training":reports["gpu-loss"]["final_training"],"final_validation":reports["gpu-loss"]["final_validation"],
            "dataset_sha256":digest(folder/"dataset.jsonl"),"model_sha256":digest(folder/"gpu-loss-0/model.json")}
        cases.append(case);write(out/"partial.json",cases)
    result={"scope":"original synthetic bounded GPU prediction/loss preparation; no physical validation or general scaling claim",
       "process_measurement":"Complete Windows subprocess wall time and PeakWorkingSetSize, including startup, validation, assessment, checkpoint/export IO; three alternating-order repetitions",
       "cases":cases,"sha256":{"script":digest(__file__),"fixture_recipe":digest(root/"tools/replay_tiled_training.py"),
       "trainer":digest(trainer),"evaluator":digest(evaluator),"historical_v7_trainer":digest(reference)}}
    write(out/"acceptance.json",result)
    print(json.dumps({"complete":True,"cases":[{k:c[k] for k in ("case","whole_process_speed_ratio","gradient_speed_ratio")}for c in cases]}),flush=True)
if __name__=="__main__":main()
