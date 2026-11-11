"""Non-optimizing real-model checks. Never invokes qualification or training.

Run with --device cpu (default), mps or cuda. Reference baseline studies are an
explicit optional CPU64 label-only check of the OLD six-case validation grid.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from pdeep.artifact_io import atomic_json
from pdeep.artifact_tensors import state_identity
from pdeep.core import require
from pdeep.learning_diagnostics.certificates import resolve_certification,SUPPLIED_ID
from pdeep.learning_diagnostics.data import prepare_data
from pdeep.learning_diagnostics.metrics import reconstruction,representation,gradient_terms
from pdeep.learning_diagnostics.sensitivity import sensitivity
from pdeep.geometric.models import create
from pdeep.geometric.physics import Adapter
from pdeep.representation_checks import sync


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--device',default='cpu');ap.add_argument('--output',default='reports/learning_diagnostics/real-model-verification');ap.add_argument('--include-reference-baselines',action='store_true');args=ap.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    calls=[]
    def forbidden(*a,**k):calls.append(1);raise AssertionError('NO optimizer construction/steps permitted')
    torch.optim.AdamW.__init__=forbidden;torch.optim.AdamW.step=forbidden
    from pdeep.smooth import qualification
    qualification.ensure_compatible_certificate=forbidden
    cert=resolve_certification(requested_capabilities={'certification_id':None},cache_dir=root/'certifications').require()
    issued=resolve_certification(cache_dir=root/'certifications')
    from pdeep.runtime import environment_report
    result={'runtime':environment_report(),'device':args.device,'supplied_certificate':issued.summary(),'execution_certificate':cert['certification_id'],'qualification_executed':False,'optimizer_calls':0,'models':[]}
    data={p:prepare_data(root/'fixed_data',cert,experiment=e,count=16,grid_size=6) for p,e in [('poisson','poisson_square'),('heat','heat_single'),('wave','wave_single')]}
    for arch in ('A_GEO','C_GEO'):
        m=create(arch,data['poisson'].normalizers,'medium',seed=42,device=args.device);before=state_identity(m)
        for pde,d in data.items():
            print(arch,pde,'forward/backward ONLY',flush=True);start=time.perf_counter();b=d.batch(args.device)
            cs=m.prepare_context(b.specs,b.observations);pred=m.predict(cs,b.queries,chunk=256);one=m.predict(cs,b.queries,chunk=1)
            scales=pred.new_tensor([d.normalizers.scale(pde,int(q)) for q in b.queries.quantity_id[0]])[None,:,None]
            chunkerr=float(((pred-one)/scales).detach().abs().max());require(chunkerr<2e-5,'query chunk mismatch')
            normalized=(pred-b.queries.value)/scales;loss=sum(normalized[0,b.queries.quantity_id[0]==q].square().mean() for q in (0,1) if (b.queries.quantity_id[0]==q).any())/(2 if pde=='wave' else 1)
            m.zero_grad(set_to_none=True);loss.backward();grad={}
            for name,param in m.named_parameters():
                if param.grad is not None:
                    require(bool(torch.isfinite(param.grad).all()),'nonfinite gradient')
                    group=name.split('.')[0];grad[group]=grad.get(group,0.)+float(param.grad.square().sum())
            grad={k:v**.5 for k,v in grad.items()};require(grad['observation']>0 and grad['decoder']>0 and grad['u_head']>0,'missing field gradient path')
            if pde=='wave':require(grad['v_head']>0,'missing wave v gradient')
            m.zero_grad(set_to_none=True)
            metrics=reconstruction(m,d);rep=representation(m,d);sense=sensitivity(m,d,query_limit=32)
            require(sense['perturbations']['complete_record_permutation']['standardized_RMS_difference']<2e-5,'permutation invariance failed')
            require(state_identity(m)==before,'non-optimizing diagnostics changed parameters')
            sync(torch.device(args.device));row={'architecture':arch,'capacity':'medium','parameters':m.parameter_count,'pde':pde,'dtype':'float32','physical_case_id':d.case.case_id,'field_gradient_norms':grad,'query_chunk_max_standardized_error':chunkerr,'metrics':metrics,'representation':rep,'sensitivity':sense,'seconds':time.perf_counter()-start,'weights_changed':False,'status':'untrained-forward/backward evidence; not learning/conditioning success'}
            result['models'].append(row);atomic_json(root/'report.json',result)
    # Full physics checks use the SAME C implementation at bounded verification width;
    # no substitute network and no optimizer. Medium full-physics training is manual.
    m=create('C_GEO',data['wave'].normalizers,'verification',seed=42,device=args.device)
    print('C_GEO wave bounded full-physics gradient check, zero updates',flush=True)
    result['bounded_full_wave_physics']=gradient_terms(m,data['wave'].batch(args.device),Adapter('C_GEO'))
    if args.include_reference_baselines:
        from pdeep.smooth.bank import grid
        from pdeep.smooth.data import Case,evaluate_pairs
        from pdeep.smooth.definition import SEEN
        provided={('A_GEO','poisson','square','u'):.17363,('A_GEO','poisson','rectangle','u'):2.4821,('A_GEO','heat','square','u'):1.3095,('A_GEO','heat','rectangle','u'):2.8669,('A_GEO','wave','square','u'):1.4812,('A_GEO','wave','rectangle','u'):1.3886,('A_GEO','wave','square','v'):.95867,('A_GEO','wave','rectangle','v'):.59929,('C_GEO','poisson','square','u'):.59698,('C_GEO','poisson','rectangle','u'):2.14,('C_GEO','heat','square','u'):.99758,('C_GEO','heat','rectangle','u'):3.5732,('C_GEO','wave','square','u'):1.5608,('C_GEO','wave','rectangle','u'):1.7701,('C_GEO','wave','square','v'):.94976,('C_GEO','wave','rectangle','v'):.58441}
        rows=[]
        for pde in ('poisson','heat','wave'):
            for family in ('square','rectangle'):
                c=Case(family,pde,'train',SEEN.index(family));x,t,q,_=grid(c,64);print('CPU64 zero baseline',pde,family,flush=True)
                yy=evaluate_pairs(c,x,t,q)
                for k in (0,1) if pde=='wave' else (0,):
                    name='v' if k else 'u';y=yy[q==k];s=data[pde].normalizers.scale(pde,name);zero=float(np.mean((y/s)**2))
                    rows.append({'pde':pde,'family':family,'quantity':name,'case_id':c.case_id,'grid':'64x64 clipped interior, existing9 evaluation times','points':len(y),'zero_baseline_standardized_MSE':zero,'truth_physical_RMS':float(np.sqrt(np.mean(y*y))),'provided_trained_MSE':{a:provided[a,pde,family,name] for a in ('A_GEO','C_GEO')},'provided_MSE_over_zero_baseline':{a:provided[a,pde,family,name]/zero for a in ('A_GEO','C_GEO')},'scope':'Truth and zero baseline independently evaluated; trained errors user-provided, trained weights unavailable here'})
        result['original_six_case_zero_baselines']=rows
    require(not calls,'optimizer/qualification was called')
    atomic_json(root/'report.json',result);print('VERIFIED; optimizer calls=0; qualification runs=0',root/'report.json',flush=True)

if __name__=='__main__':main()
