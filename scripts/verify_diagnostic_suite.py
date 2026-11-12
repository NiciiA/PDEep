"""Bounded real suite-path forward/backward verification; ZERO optimizer steps.

Uses the existing native-runtime numerical certificate and full certified P4
references. No training runner, new qualification, or neural checkpoint is loaded.
CUDA/Drive are only claimed when actually present; local filesystem tests are
reported separately by pytest. Optional --chunks measures four same-Q decoders.
"""
import argparse
import gc
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from pdeep.artifact_io import atomic_json
from pdeep.artifact_tensors import state_identity
from pdeep.core import require
from pdeep.learning_diagnostics.certificates import resolve_certification
from pdeep.learning_diagnostics.suite import prepare_diagnostic_suite,chunk_preflight
from pdeep.learning_diagnostics.workflows import prepare_diagnostic,performance_preflight
from pdeep.learning_diagnostics.queries import QuerySampler
from pdeep.learning_diagnostics.query_loss import QueryAdapter
from pdeep.geometric.physics import WeakConfig
from pdeep.learning_diagnostics.performance import profile


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',default='cpu');parser.add_argument('--output',default='reports/diagnostic_suite/real-cpu');parser.add_argument('--chunks',action='store_true');parser.add_argument('--experiments',nargs='+',default=['poisson_rectangle','heat_single','wave_single']);args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    certificate=resolve_certification(requested_capabilities={'certification_id':None},cache_dir=root/'certifications').require()
    def forbidden(*a,**kw):raise AssertionError('ZERO optimizer steps / qualifications permitted')
    for cls in (torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD):cls.step=forbidden
    import pdeep.smooth.qualification as qualification
    qualification.ensure_compatible_certificate=forbidden
    suite=prepare_diagnostic_suite(args.experiments,certificate=certificate,root=root,settings={'device':args.device})
    result={'preview':suite.preview(),'device':args.device,'CUDA_available':torch.cuda.is_available(),'actual_CUDA_measurement':args.device.startswith('cuda'),
            'actual_Google_Drive_tested':False,'optimizer_updates':0,'qualification_runs':0,'rows':[]}
    for recipe in suite.recipes:
        print('Preparing certified fixed reference:',recipe['experiment'],flush=True)
        p=prepare_diagnostic(model='A_GEO',capacity='medium',experiment=recipe['experiment'],updates=1000,device=args.device,seed=recipe['seed'],certificate=certificate,root=root,
            query_mode='dense_resampled',query_count=8192,query_time_policy='stratified',query_sampler_seed=42,query_chunk=1024,query_checkpointing=False,
            amp_mode='auto',query_revision='continuous-queries-v3',evaluation_times=(0.,.05,.1,.25,.5,.75,1.),advanced={'grid_size':6})
        before=state_identity(p.models[0]);perf=performance_preflight(p)
        sampler=QuerySampler(p.data,p.configs[0],reference_cache=p.reference_cache);solves_before=sampler.reference.solves;a,info=sampler.batch(0);b,other=sampler.batch(1)
        require(info['query_identity']!=other['query_identity'] and info['observation']==other['observation'] and info['physical_case']==other['physical_case'],'resampling/fixed conditioning failure')
        require(sampler.reference.solves==solves_before,'reference performed new solves during query resampling')
        q=a.queries;coverage={}
        for quantity in q.quantity_id.unique().tolist():
            t=q.physical_time[0,q.quantity_id[0]==quantity,0].detach().cpu().numpy()
            coverage['v' if quantity else 'u']={'count':len(t),'zero':int((t==0).sum()),'early':int(((t>0)&(t<=.1)).sum()),'middle':int(((t>.1)&(t<=.5)).sum()),'late':int(((t>.5)&(t<=1)).sum())}
        adapter=QueryAdapter('A_GEO',WeakConfig(),chunk=1024,checkpointing=False,amp=p.configs[0].resolved_amp_mode)
        loss=adapter.loss(p.models[0],a);loss.total.backward();terms=adapter.last['per_quantity_supervision'][0]
        weighted=sum(v['weighted_contribution'] for v in terms.values());discrepancy=abs(float(loss.unweighted['field'].detach())-weighted)
        require(discrepancy<=2e-6*max(1,weighted),'per-quantity reduction differs')
        grads={name.split('.')[0]:0. for name,v in p.models[0].named_parameters() if v.grad is not None}
        for name,v in p.models[0].named_parameters():
            if v.grad is not None:require(torch.isfinite(v.grad).all(),'nonfinite gradient');grads[name.split('.')[0]]+=float(v.grad.square().sum())
        require(grads['u_head']>0 and (p.data.case.pde!='wave' or grads['v_head']>0),'missing output-head derivative path')
        p.models[0].zero_grad(set_to_none=True)
        row={'experiment':recipe['experiment'],'parameters':p.models[0].parameter_count,'performance':perf,'coverage':coverage,'per_quantity_loss':terms,'loss_reduction_max_discrepancy':discrepancy,
            'gradient_squared_norms':grads,'frozen_observation':info['observation'],'frozen_case':info['physical_case'],'query_identities':[info['query_identity'],other['query_identity']],
            'reference_preparation_linear_solves':sampler.reference.solves,'new_reference_solves_during_resampling':sampler.reference.solves-solves_before,'reference_bytes':sampler.reference.bytes,'reference_evaluations':[info,other],
            'evaluation_times':sorted(set(p.data.diagnostic.physical_time.flatten().tolist()))}
        if args.chunks and recipe['experiment']=='poisson_rectangle':row['chunk_sweep']=chunk_preflight(p)
        require(state_identity(p.models[0])==before,'nontraining verifier changed weights')
        result['rows'].append(row);atomic_json(root/'verification.json',result)
        del loss,adapter,a,b,sampler,p;gc.collect()
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
        elif args.device=='mps':torch.mps.empty_cache()
    print('Verified real forward/backward only; ZERO optimizer updates.',flush=True)

if __name__=='__main__':main()
