"""Fixed-work, three-implementation process comparison. Python stdlib only.
Requires an authorized exclusive GPU window and prepared public/synthetic inputs.
Preserves all labels, updates, precision, checkpoint cadence and export validation.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
from replay_full_training import digest, write
from replay_indexed_training import launch
from replay_transfer_batching import numeric_error


def span(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-exe", "transfer-exe", "current-exe", "prepared-dir", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--cases", nargs="+", default=["physical","medium","tiny"])
    a=p.parse_args(); prepared=a.prepared_dir.resolve(); out=a.output_dir.resolve();out.mkdir(exist_ok=False)
    exes={"baseline":a.baseline_exe.resolve(),"transfer":a.transfer_exe.resolve(),"current":a.current_exe.resolve()}
    result={"method":"Three fixed rotating/alternating GPU implementations, three fresh processes each; two CPU reference processes bracketing them. No precision/work/output changes. Startup is fresh-process NVRTC/driver initialization with existing uncontrolled caches, not a claimed cold driver-cache experiment.",
        "executables_sha256":{k:digest(v) for k,v in exes.items()},"executed_script_sha256":digest(__file__),"cases":[]}
    for name in a.cases:
        folder=prepared/name;req=json.loads((folder/"profile.request.json").read_text())
        req["dataset"]=str(folder/"dataset.jsonl");req["initial_model"]=str(folder/"initial.json")
        rows=[];states={};reports={}
        def execute(label, exe, request):
            r=deepcopy(request);dest=out/(name+"-"+label);r["output_directory"]=str(dest)
            path=out/(name+"-"+label+".request.json");write(path,r);process=launch(exe,path)
            state=json.loads((dest/"checkpoint.json").read_text());report=json.loads((dest/"report.json").read_text())
            assert report["native_v1_reload_max_abs_energy_force_virial"]==[0.,0.,0.]
            row={"label":label,"process":process,"timing":report["timing"],"profile":report.get("cuda_gradient_profile"),
                "state_profile":report.get("cuda_state_residency"),"output_timing":report.get("output_timing"),
                "checkpoint_sha256":digest(dest/"checkpoint.json"),"model_sha256":digest(dest/"model.json")}
            rows.append(row);print(json.dumps({"case":name,"run":label,"process_ms":process["process_ms"],"gradient_ms":report["timing"]["gradient_ms"],"adam_ms":report["timing"]["adam_ms"]}),flush=True)
            write(out/"progress.json",{"completed_cases":result["cases"],"active_case":name,"runs":rows})
            return state,report
        cpu=deepcopy(req);cpu["device"]="cpu"
        for k in ("cuda_gradient","cuda_tiling","cuda_loss","fixed_geometry_cache"):cpu["config"].pop(k,None)
        cpus,cpur=execute("cpu-0",exes["current"],cpu)
        for repeat,order in enumerate((("baseline","transfer","current"),("current","transfer","baseline"),("transfer","baseline","current"))):
            for mode in order:
                state,report=execute(f"{mode}-{repeat}",exes[mode],req)
                prior=[r for r in rows[:-1] if not r["label"].startswith("cpu")]
                if prior:
                    assert rows[-1]["checkpoint_sha256"]==prior[0]["checkpoint_sha256"]
                    assert rows[-1]["model_sha256"]==prior[0]["model_sha256"]
                    old=next(iter(reports.values()))
                    for k in ("execution","final_training","final_validation","training_frames","validation_frames"):
                        assert report[k]==old[k],k
                    for k in ("frames","tiles","kernel_launches","synchronizations"):
                        assert report["cuda_gradient_profile"][k]==old["cuda_gradient_profile"][k]
                    assert report["cuda_tiling"]["effective_max_atoms"]==old["cuda_tiling"]["effective_max_atoms"]
                if mode=="current":
                    stat=report["cuda_state_residency"];steps=req["config"]["steps"];interval=req["config"]["checkpoint_every"]
                    assert stat["updates"]==steps and stat["status_downloads"]==steps
                    assert stat["snapshots"]==(steps+interval-1)//interval
                    assert stat["state_download_calls"]==3*stat["snapshots"]
                    assert report["output_timing"]["reload_evaluations"]==2*(len(report["training_frames"])+len(report["validation_frames"]))
                    assert report["cuda_gradient_profile"]["peak_device_buffer_bytes"]<=64*1024*1024
                    assert report["cuda_gradient_profile"]["peak_host_tile_payload_bytes"]<=32*1024*1024
                states[mode]=state;reports[mode]=report
        cs2,cr2=execute("cpu-1",exes["current"],cpu)
        assert cpus==cs2
        error=numeric_error(cpus,states["current"]);assert error<1e-10
        assert cpus["sampler"]==states["current"]["sampler"]
        summaries={}
        for mode in ("baseline","transfer","current","cpu"):
            rr=[r for r in rows if r["label"].startswith(mode+"-")]
            summaries[mode]={"runs":len(rr),"whole_process_ms":span([r["process"]["process_ms"] for r in rr]),
                "startup_ms":span([r["timing"]["startup_ms"] for r in rr]),
                "steady_updates_ms":span([r["timing"]["gradient_ms"]+r["timing"]["adam_ms"] for r in rr]),
                "gradient_ms":span([r["timing"]["gradient_ms"] for r in rr]),"adam_ms":span([r["timing"]["adam_ms"] for r in rr])}
        case={"case":name,"steps":req["config"]["steps"],"config":req["config"],"dataset_access":req["dataset_access"],
            "dataset_sha256":digest(folder/"dataset.jsonl"),"initial_model_sha256":digest(folder/"initial.json"),
            "summary":summaries,"runs":rows,"all_gpu_checkpoint_and_model_files_byte_identical":True,
            "execution":reports["current"]["execution"],"cpu_model_and_moment_max_abs":error,"cpu_repeats_exact":True,
            "baseline_over_current_whole":summaries["baseline"]["whole_process_ms"]["median"]/summaries["current"]["whole_process_ms"]["median"],
            "transfer_over_current_whole":summaries["transfer"]["whole_process_ms"]["median"]/summaries["current"]["whole_process_ms"]["median"],
            "cpu_over_current_whole":summaries["cpu"]["whole_process_ms"]["median"]/summaries["current"]["whole_process_ms"]["median"]}
        result["cases"].append(case);write(out/"acceptance.json",result)
        print(json.dumps({"case":name,"summary":summaries}),flush=True)
    print("Resident state process comparison complete",flush=True)


if __name__=="__main__":main()
