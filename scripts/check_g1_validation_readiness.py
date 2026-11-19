"""Bounded CPU verification on an EXISTING dataset; no optimizer or PDE solves.

.venv/bin/python scripts/check_g1_validation_readiness.py --manifest EXACT/manifest.json
CPU verification-capacity models only: not a large-model CUDA memory claim.
"""
import time,json,torch,numpy as np
from pathlib import Path
from pdeep.learning_diagnostics.certificates import resolve_certification
from pdeep.preparation.loading import load_dataset
from pdeep.preparation.feasibility_training import configurations
from pdeep.preparation.readiness import ensure_ready
from pdeep.preparation import coordinates
from pdeep.preparation.validation import evaluate
from pdeep.unified_modes.model import create
from pdeep.unified.streams import coordinates_id
from pdeep.unified.config import SCHEMA
from pdeep.smooth.reference import Reference
from pdeep.artifact_rng import capture_rng
from pdeep.artifact_tensors import tree_identity

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest',required=True)
parser.add_argument('--root',default='reports/g1_validation_readiness')
args=parser.parse_args()
torch.set_num_threads(1)
def forbidden(*a,**k):raise AssertionError('online PDE solve')
Reference.__init__=forbidden
path=Path(args.manifest)
cert=resolve_certification(requested_capabilities={'certification_id':None}).require();d=load_dataset(path,cert);root=Path(args.root);cfg=configurations(d.config,device='cpu',experiment_id='audit-only')
t=time.perf_counter();r=ensure_ready(cfg,d,root,progress=lambda *a,**k:None,force=True);cold=time.perf_counter()-t
t=time.perf_counter();ensure_ready(cfg,d,root,progress=lambda *a,**k:None);warm=time.perf_counter()-t
native=coordinates.physical_design
# Reproduce the exact reported consumer bytes, then require exact producer SHA.
def separate(case,n,**kw):
 design=native(case,n,**kw)
 return next(design for method,design in coordinates.arithmetic_variants(case,n,**kw,design=design) if method=='separate-012')
case=next(c for c in d.cases('validation') if c.pde=='poisson' and c.family=='square')
sid=('unified-validation',d.config.seed);reported=coordinates_id(*separate(case,512,set_id=sid,observation=False)[:3])
if d.config.seed==42:assert reported=='fbdad2015f317547a38533ce51c1985c3a24e6cb1d249c418f0bb3cce38f0e4b'
coordinates.physical_design=separate;t=time.perf_counter();checked=0
for c in d.cases('validation'):
 for n,obs,s in ((512,False,('unified-validation',d.config.seed)),(64,True,(SCHEMA,'fixed-evaluation',d.config.seed))):
  design=coordinates.prepared_design(d,c,n,observation=obs,set_id=s)
  assert coordinates_id(*design[:3]) in d.cache._payload(c)['labels'];y=d.cache.evaluate(c,*design[:3]);assert len(y)==n and np.isfinite(y).all();checked+=1
replay=time.perf_counter()-t;coordinates.physical_design=native
rows=[];identities=[]
for key in ('A_GEO/KIN_001','C_GEO/FIELD_ONLY','C_GEO/ALL_001'):
 net=create(key.split('/')[0],d.norms,'verification',seed=42,device='cpu');state=tree_identity(capture_rng());t=time.perf_counter();result=evaluate(net,d,model_key=key);seconds=time.perf_counter()-t
 assert state==tree_identity(capture_rng())
 identity={(r['case_id'],r['mode'],r['quantity']):(r['query_identity'],r['observation_identity'],r['field_condition_identity']) for r in result['rows']}
 assert len(result['rows'])==192 and len({(r['pde'],r['family']) for r in result['rows']})==24 and set(r['split'] for r in result['rows'])=={'validation'}
 identities.append(identity);rows.append({'model':key,'capacity':'verification (NOT large GPU memory test)','seconds':seconds,'rows':len(result['rows']),'physical_cases':48,'information_views':144,'optimizer_updates':0});print(rows[-1],flush=True)
assert identities[0]==identities[1]==identities[2]
report={'status':'ready','checked_designs':r['checked_designs'],'training_designs':24000,'validation_designs':96,'missing':0,'complete_audit_seconds':cold,'cached_audit_seconds':warm,'reported_consumer_hash':reported,'cross_CPU_validation_designs':checked,'cross_CPU_replay_seconds':replay,'shared_validation':True,'validation_models':rows,'reference_solves':0,'manifest_identity':d.index['manifest_identity'],'actual_CUDA_test':False,'actual_Drive_test':False}
(root/'verified_workflow.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
