"""Export exact existing meshes/prior bases and bounded CPU64 comparison fixtures.

Run BEFORE portability implementation changes. No training/model/optimizer calls.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
from pdeep.core import digest
from pdeep.smooth.definition import *
from pdeep.smooth.reference import Reference
from pdeep.wave_fidelity.fields import centers
OUT=Path('certifications/G1_smooth_v1/portable');OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(shape,h):return digest({'geometry':shape.geometry_id,'rings':[len(shape.outer),*[len(r) for r in shape.holes]],'tags':list(shape.tags),'h':float(h)})
def export(family):
 start=time.perf_counter();files={};meshes={};canonical=geometry(family,.08)
 for h in (.01,.02,.04,.08,.12):
  m=geometry(family,h);name=f'{family}-h{h:g}.npz';p=OUT/name
  if not p.exists():m.save(p)
  files[name]=sha(p);meshes[key(m.shape,h)]={'file':name,'mesh_id':m.mesh_id,'geometry_id':m.geometry_id,'polygon_id':m.shape.polygon_id,'h':h,'sha256':sha(p)}
 c,radius=centers(canonical);basis=OUT/(family+'-basis.npz')
 if not basis.exists():np.savez_compressed(basis,centers=c,radius=radius)
 files[basis.name]=sha(basis)
 x=canonical.sample(128,random_stream('certification',family,'field-coordinates'));sensors=canonical.sample(128,random_stream('certification',family,'sensor-coordinates'))
 st=random_stream('certification',family,'continuous-sensor-times').random(128);st[:8]=0.;st[8:24]*=.1
 xx=np.concatenate((np.tile(x,(len(TIMES),1)),sensors));tt=np.concatenate((np.repeat(TIMES,len(x)),st));cut=len(x)*len(TIMES)
 arrays={'position':xx,'time':tt};ids={};split='validation' if family in SEEN else 'geometry_held_out';r=Reference(family,.02,4)
 for p in PDES:
  print(f'export {family}/{p}: four fixed fields, production CPU64',flush=True);fields=[physical_fields(family,p,split,i) for i in range(4)];ids[p]=[{k:v.identity for k,v in f.items()} for f in fields]
  if p=='poisson':u,_=r.poisson([f['f'] for f in fields],xx);v=None
  else:u,v,_=r.dynamic(p,[f['u0'] for f in fields],[f['v0'] for f in fields] if p=='wave' else None,xx,tt,.002)
  arrays[p+'_u']=u
  if v is not None:arrays[p+'_v']=v
  old=Path('reports/smooth_finalization')/(family+'-values.npz')
  if old.exists():
   expected=np.load(old);delta=float(abs(u[0,:cut].reshape(len(TIMES),len(x))-expected['production_'+p+'_u']).max());assert delta<1e-13,(family,p,delta)
 fixture=OUT/(family+'-reference.npz')
 if not fixture.exists():np.savez_compressed(fixture,**arrays)
 files[fixture.name]=sha(fixture)
 body={'family':family,'meshes':meshes,'basis_file':basis.name,'basis_sha256':sha(basis),'reference_file':fixture.name,'reference_sha256':sha(fixture),'field_identities':ids,'files':files,'seconds':time.perf_counter()-start,'parent_certification_id':json.loads(Path('certifications/G1_smooth_v1/default.json').read_text())['record'].removesuffix('.json'),'source_revision':physics_identity()}
 (OUT/(family+'.json')).write_text(json.dumps(body,indent=2)+'\n');print('DONE',family,body['seconds'],flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('families',nargs='+');args=parser.parse_args()
 for family in args.families:export(family)
