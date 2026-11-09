"""Bounded real-model capability/fixed-validation checks; ZERO optimizer updates.

The notebook owns the sole Stage7A three-update-per-variant preflight. This script
checks additional devices without exceeding that scientific training budget.
"""
from dataclasses import asdict
from pathlib import Path
import argparse,json,uuid
from pdeep.data import calibrate
from pdeep.diagnostics import source_identity
from pdeep.runtime import capabilities,environment_report
from pdeep.train_types import ResolvedTrainConfig
from pdeep.training import TrainingEngine


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    p.add_argument('--architectures',nargs='+',choices=('A','C','B0','B1'),default=['A','C','B0','B1']);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];norm=calibrate();source=source_identity(root)['sha256'];tag='capability-'+args.device+'-'+uuid.uuid4().hex[:10]
    records={}
    for name in args.architectures:
        cfg=ResolvedTrainConfig(name,device=args.device,source_id=source,run_tag=tag,microbatch=1 if name=='C' else 2,accumulation=3 if name=='C' else 2,effective_batch=3,query_count=8,validation_query_count=8)
        engine=TrainingEngine(cfg,norm,root/'artifacts/training_runs')
        validation=engine.validate();assert engine.state.successful==0
        records[name]={'capability':engine.capability,'validation':asdict(validation),'optimizer_updates':0}
    directory=root/'reports/training_engine'/tag;directory.mkdir(parents=True,exist_ok=True);path=directory/'capability_validation.json'
    path.write_text(json.dumps({'source_sha256':source,'device':args.device,'capabilities':[asdict(c) for c in capabilities()],'environment':environment_report(),'records':records},indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'device':args.device,'architectures':args.architectures,'optimizer_updates':0,'report':str(path)},indent=2))


if __name__=='__main__':main()
