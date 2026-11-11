"""Bounded independent-coordinate performance check; never launches comparison arms.

Optional --optimizer-smoke makes ONE disposable cloned-model AdamW step for the
whole invocation. Original model, physical case and checkpoints are never updated.
CUDA memory/throughput is reported only if actually executed on CUDA.
"""
import argparse
from pathlib import Path
import torch
from pdeep.artifact_io import atomic_json
from pdeep.core import require
from pdeep.artifact_tensors import state_identity
from pdeep.learning_diagnostics.certificates import resolve_certification
from pdeep.learning_diagnostics.data import prepare_data
from pdeep.learning_diagnostics.engine import DiagnosticConfig
from pdeep.learning_diagnostics.queries import QuerySampler
from pdeep.learning_diagnostics.query_loss import QueryAdapter
from pdeep.learning_diagnostics.performance import profile,resolve_amp
from pdeep.geometric.models import create
from pdeep.geometric.physics import WeakConfig


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');p.add_argument('--counts',default='1024,2048,4096,8192');p.add_argument('--optimizer-smoke',action='store_true');p.add_argument('--output',default='reports/query_diagnostics/performance-cpu');p.add_argument('--checkpointing',action='store_true');args=p.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    cert=resolve_certification(requested_capabilities={'certification_id':None},cache_dir=root/'certifications').require()
    from pdeep.smooth import qualification
    qualification.ensure_compatible_certificate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No qualification allowed'))
    data=prepare_data(root/'fixed-data',cert,count=1024,grid_size=6,cache_reference=True)
    report={'device':args.device,'CUDA_available':torch.cuda.is_available(),'actual_CUDA_measurement':args.device.startswith('cuda'),'qualification_runs':0,'research_training_updates':0,'disposable_optimizer_steps':0,'rows':[],'case_identity':data.case.case_id,'observations':data.identities['observation'],'independent':data.identities['diagnostic']}
    ref=data.cached_reference
    for arch in ('A_GEO','C_GEO'):
        m=create(arch,data.normalizers,'medium',seed=42,device=args.device);before=state_identity(m)
        for n in [int(v) for v in args.counts.split(',')]:
            cfg=DiagnosticConfig(architecture=arch,capacity='medium',updates=1,device=args.device,pdes=('poisson',),probabilities=(1.,),training_families=('square',),cases_per_geometry=1,query_count=n,bank_queries=n,data_identity=data.identity,query_revision='continuous-queries-v2',query_mode='dense_resampled' if n>1024 else 'resampled',query_checkpointing=args.checkpointing,query_chunk=1024)
            sampler=QuerySampler(data,cfg,reference_cache=ref);sampler.measure_transfer=True;b,metadata=sampler.batch(0);ref=sampler.reference
            adapter=QueryAdapter(arch,WeakConfig(),chunk=1024,checkpointing=args.checkpointing)
            precision=resolve_amp(m,b,adapter,'auto');adapter.amp=precision['resolved']
            smoke=args.optimizer_smoke and len(report['rows'])==0
            print(arch,n,'isolated clone; optimizer step',smoke,flush=True)
            result=profile(m,b,adapter,optimizer_step=smoke)
            result.update(architecture=arch,capacity='medium',parameters=m.parameter_count,precision_policy=precision,data=metadata)
            report['rows'].append(result);report['disposable_optimizer_steps']+=result['optimizer_steps'];atomic_json(root/'performance.json',report)
            require(state_identity(m)==before,'original model was changed')
            if result['status']!='fits':break
    print('Verified; disposable optimizer steps',report['disposable_optimizer_steps'],'research updates0; CUDA measured',report['actual_CUDA_measurement'],flush=True)

if __name__=='__main__':main()
