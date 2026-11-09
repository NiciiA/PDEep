"""Bounded B0/B1 mathematical/path checks; no automatic optimizer or ablation training."""
import argparse,json
from pathlib import Path
from pdeep.b_checks import inverse_math_checks,architecture_b_checks
from pdeep.data import calibrate
from pdeep.core import digest
from pdeep.diagnostics import source_identity
from pdeep.runtime import resolve_device,environment_report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu');args=p.parse_args();resolve_device(args.device,'float32')
    norm=calibrate();math=inverse_math_checks(norm);checks=architecture_b_checks(norm,device=args.device)
    if not math['passed'] or not checks['passed']:raise RuntimeError('B verification failed')
    root=Path(__file__).resolve().parents[1];source=source_identity(root);run=digest(dict(source=source['sha256'],device=args.device,version='B-inverse-v1'))
    directory=root/'reports/architecture_B'/run;directory.mkdir(parents=True,exist_ok=True)
    report=dict(run_id=run,source=source,environment=environment_report(),mathematics=math,checks=checks,training=None)
    path=directory/'verification.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,device=args.device,parameters={k:v['total_parameters'] for k,v in checks['variants'].items()},report=str(path)),indent=2))


if __name__=='__main__':main()
