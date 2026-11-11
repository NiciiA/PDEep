"""Read-only historical export compatibility; zero optimizer operations."""
from pathlib import Path
import argparse,json,torch
from pdeep.inference_artifacts import load_inference,unpack_observation,unpack_query
from pdeep.reference import canonical_spec
from pdeep.core import ValidationError
p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','mps'],required=True);a=p.parse_args();rows=[]
def forbidden(*args,**kw):raise AssertionError('G3 export verification forbids optimizer operations')
torch.optim.AdamW.step=forbidden;torch.optim.Adam.step=forbidden
for name in ('A','B0'):
 paths=sorted((Path('artifacts/training_runs')/name).glob('*/inference/*.json'))
 path=next(p for p in paths if json.loads(p.read_text()).get('successful_updates',0)>=10000)
 error=None
 try:wrapper=load_inference(path,expected_architecture=name,device=a.device)
 except ValidationError as e:
  error=str(e)
  # Diagnostic only: weights/schema/checksums remain verified; do NOT relax or
  # overwrite the production export's declared example tolerance.
  wrapper=load_inference(path,expected_architecture=name,device=a.device,verify_examples=False)
 checks=[];tol=wrapper.metadata['example_tolerance']
 with torch.no_grad():
  for e in wrapper.payload['examples']:
   ctx=wrapper.prepare(canonical_spec(e['pde']),unpack_observation(e['observations']));v=wrapper.predict(ctx,unpack_query(e['queries']),chunk=3);expected=e['expected_prediction']
   checks.append({'pde':e['pde'],'max_prediction_error':float((v-expected).abs().max()),'max_xi_error':float((ctx.xi-e['expected_xi']).abs().max()),'finite':bool(torch.isfinite(v).all()),'within_stored_tolerance':bool(torch.allclose(v,expected,**tol))})
 rows.append({'architecture':name,'path':str(path),'device':a.device,'recorded_training_device':wrapper.metadata['training_config']['device'],'artifact_id':wrapper.metadata['artifact_id'],
              'new_optimizer_updates':0,'strict_example_load_passed':error is None,'strict_error':error,'stored_tolerance':tol,'examples':checks})
 print(name,a.device,'strict load:',error is None,'max error',max(c['max_prediction_error'] for c in checks),flush=True)
Path('reports/wave_fidelity_g3/T1-exports-'+a.device+'.json').write_text(json.dumps(rows,indent=2))
