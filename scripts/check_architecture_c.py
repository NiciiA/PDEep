"""Bounded C checks; optional explicit 3-update smoke, no large training."""
import argparse,json
from dataclasses import asdict
from pathlib import Path
from pdeep.c_math_checks import mathematical_constraint_checks
from pdeep.c_checks import architecture_c_checks
from pdeep.c_training import CTrainingConfig,run_bounded
from pdeep.architecture_c import build_architecture_c
from pdeep.c_artifacts import save_inference
from pdeep.data import calibrate,CaseDataset
from pdeep.diagnostics import source_identity
from pdeep.core import digest
from pdeep.runtime import resolve_device,environment_report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--tiny-overfit',type=int,default=0)
    args=parser.parse_args();resolve_device(args.device,'float32')
    if args.smoke and args.tiny_overfit:parser.error('choose smoke OR tiny-overfit')
    config=CTrainingConfig(preset='tiny_overfit',tiny_overfit=True,updates=args.tiny_overfit) if args.tiny_overfit else CTrainingConfig()
    math=mathematical_constraint_checks()
    if not math['passed']:raise RuntimeError('mandatory mathematical constraints checks failed')
    norm=calibrate();checks=architecture_c_checks(norm,device=args.device)
    if not checks['passed']:raise RuntimeError('C structure/gradient checks failed')
    root=Path(__file__).resolve().parents[1];source=source_identity(root)
    run=digest(dict(source=source['sha256'],device=args.device,smoke=args.smoke,tiny=args.tiny_overfit))
    directory=root/'reports/architecture_C'/run;directory.mkdir(parents=True,exist_ok=True)
    report=dict(run_id=run,source=source,environment=environment_report(),mathematics=math,checks=checks,training=None,config=asdict(config),normalizer_id=norm.identity,dataset_id=CaseDataset().identity)
    if args.smoke or args.tiny_overfit:
        c=build_architecture_c(norm).to(args.device)
        report['training'],_=run_bounded(c,CaseDataset(),norm,config)
        report['inference_manifest']=str(save_inference(c,root/'artifacts/C_inference',source=source['sha256'],config=asdict(config),status=report['training']['status'],dataset_id=CaseDataset().identity).relative_to(root))
    path=directory/'verification.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (root/'reports/latest_architecture_C.json').write_text(json.dumps(dict(run_id=run,report=str(path.relative_to(root)),directory=str(directory.relative_to(root))),indent=2)+'\n')
    print(json.dumps(dict(passed=True,parameters=checks['total_parameters'],device=args.device,training=report['training']['counters'] if report['training'] else None,report=str(path)),indent=2))


if __name__=='__main__':main()
