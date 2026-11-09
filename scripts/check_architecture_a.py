"""Bounded A verification; structural checks by default, 3-update smoke opt-in.

Notebook explicitly runs the authorized smoke; no research/full training option.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from pdeep.a_checks import architecture_a_checks,mixed_fixture
from pdeep.a_training import ATrainingConfig,run_bounded
from pdeep.a_artifacts import save_inference,load_inference
from pdeep.architecture_a import build_architecture_a
from pdeep.core import digest
from pdeep.data import calibrate,CaseDataset
from pdeep.diagnostics import source_identity
from pdeep.runtime import environment_report,resolve_device


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("cpu","mps","cuda"),default="cpu")
    parser.add_argument("--smoke",action="store_true",help="explicitly run 3 optimizer updates")
    parser.add_argument("--tiny-overfit",type=int,default=0,help="explicit opt-in, 1–100 fixed-case updates instead of smoke")
    args=parser.parse_args();resolve_device(args.device,"float32")
    if args.smoke and args.tiny_overfit:parser.error("choose smoke OR tiny overfit")
    root=Path(__file__).resolve().parents[1];norm=calibrate();source=source_identity(root)
    config=ATrainingConfig(preset="tiny_overfit",tiny_overfit=True,updates=args.tiny_overfit) if args.tiny_overfit else ATrainingConfig()
    checks=architecture_a_checks(norm,device=args.device)
    if not checks["passed"]:raise RuntimeError("A acceptance failed")
    report={"version":"A-direct-v1","source":source,"environment":environment_report(),"checks":checks,
            "training":None,"config":asdict(config),"normalizer_id":norm.identity,"dataset_id":CaseDataset().identity}
    run=digest({"source":source["sha256"],"device":args.device,"smoke":args.smoke,"tiny_overfit":args.tiny_overfit})
    directory=root/"reports/architecture_A"/run;directory.mkdir(parents=True,exist_ok=True)
    plots=[]
    if args.smoke or args.tiny_overfit:
        a=build_architecture_a(norm).to(args.device)
        report["training"],_=run_bounded(a,CaseDataset(),norm,config)
        export=save_inference(a,root/"artifacts/A_inference",source=source["sha256"],config=asdict(config),status=report["training"]["status"],dataset_id=CaseDataset().identity)
        loaded,_=load_inference(export,device=args.device);batch=mixed_fixture(norm,args.device,queries=16)
        with torch.no_grad():
            before=a(batch.specs,batch.observations,batch.queries).predicted_value
            after=loaded(batch.specs,batch.observations,batch.queries).predicted_value
        report["reload_max_abs_error"]=float((before-after).abs().max());report["inference_manifest"]=str(export.relative_to(root))
        from pdeep.a_plotting import architecture_diagram,memory_summary,field_examples,training_curve
        import matplotlib.pyplot as plt
        for name,figure in (("architecture",architecture_diagram()),("memory_attention",memory_summary(a,norm,status=report["training"]["status"])),
                            ("fields",field_examples(a,norm,status=report["training"]["status"])),("training",training_curve(report["training"]))):
            path=directory/f"{name}.png";figure.savefig(path,dpi=110);plt.close(figure);plots.append(str(path.relative_to(root)))
    report["plots"]=plots;report["run_id"]=run
    path=directory/"verification.json";path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    (root/"reports/latest_architecture_A.json").write_text(json.dumps({"run_id":run,"directory":str(directory.relative_to(root)),"report":str(path.relative_to(root))},indent=2)+"\n")
    print(json.dumps({"passed":checks["passed"],"device":args.device,"parameters":checks["total_parameters"],"report":str(path),
                      "updates":report["training"]["counters"] if report["training"] else None},indent=2))


if __name__=="__main__":main()
