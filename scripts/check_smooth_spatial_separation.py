"""Separate fixed-dt spatial error from temporal error without resampling.

The original immutable certificate stores all4 case A-B and A-C errors.
For all4 cases compute conservative C-B bounds by triangle inequality. For case0
also use retained evaluated arrays to measure C-B directly at all128x9 points.
No new propagation, field sampling, optimization or temporal interpolation.
"""
from pathlib import Path
import json
import numpy as np
from pdeep.core import require,digest
from pdeep.smooth.definition import *
from pdeep.smooth.certification import load_record,fixed_time_spatial_bounds,THRESHOLD
from pdeep.smooth.data import Normalizers
record=load_record();out=Path('reports/smooth_finalization');norms=Normalizers.fromdict(record['normalizers']);bounds=[];measured=[]
for family in GEOMETRIES:
 arrays=np.load(out/(family+'-values.npz'))
 for p in PDES:
  bounds+=fixed_time_spatial_bounds(record,p,family)
  for q in ('u','v') if p=='wave' else ('u',):
   a=arrays[f'{"production" if p=="poisson" else "temporal_fine"}_{p}_{q}'];b=arrays[f'independent_{p}_{q}'];d=(a-b)/norms.scale(p,q)
   actual={'family':family,'pde':p,'quantity':q,'index':0,'fixed_dt':None if p=='poisson' else .001,'field_RMSE':float(np.sqrt(np.mean(d*d))),'time_RMSE':{str(t):float(np.sqrt(np.mean(d[j]**2))) for j,t in enumerate(TIMES)}}
   bound=next(v for v in bounds if (v['family'],v['pde'],v['quantity'],v['index'])==(family,p,q,0));require(actual['field_RMSE']<=bound['field_RMSE_upper_bound']+1e-12,'spatial triangle bound violated')
   measured.append(actual)
maximum=max(v['maximum_upper_bound'] for v in bounds);require(maximum<=THRESHOLD,'fixed-time spatial bound exceeds numerical budget; do not train')
result={'passed':True,'parent_certification_id':record['certification_id'],'maximum_conservative_fixed_time_spatial_RMSE_bound':maximum,'bound_cases_per_PDE_geometry':4,'actual_array_cases_per_PDE_geometry':1,'meaning':'A=P4h.02dt.002,C=P4h.02dt.001,B=P3h.01dt.001; norm(C-B)<=norm(A-B)+norm(A-C). Bound is not a measured difference. For case0 C-B also measured directly.','bounds':bounds,'actual_case0':measured}
(out/'fixed-time-spatial.json').write_text(json.dumps(result,indent=2)+'\n');print('Maximum conservative fixed-time spatial RMS bound:',maximum,'; threshold',THRESHOLD)

body={**result,'benchmark_id':BENCHMARK_ID,'source_revision':physics_identity()}
body['supplement_id']=digest(body)
base=Path('certifications/G1_smooth_v1');path=base/('spatial-'+body['supplement_id']+'.json');data=json.dumps(body,indent=2)+'\n'
if path.exists():require(path.read_text()==data,'immutable spatial supplement collision')
else:
 with path.open('x') as stream:stream.write(data)
(base/'spatial_checks.json').write_text(json.dumps({'parent_certification_id':record['certification_id'],'record':path.name},indent=2)+'\n')
