"""Independent affine D/N/mixed controls on every represented polygon."""
import json
from pathlib import Path
import numpy as np
from pdeep.core import require
from pdeep.smooth.definition import GEOMETRIES,geometry,physics_identity
from pdeep.geometric.fem import FEM,BoundaryData
rows=[]
for family in GEOMETRIES:
 m=geometry(family,.12);tags=sorted(map(int,np.unique(m.boundary_tags)));u=lambda x,t:x[:,0]+.3*x[:,1];flux=lambda x,t,n:n[:,0]+.3*n[:,1];truth=u(m.vertices,0)
 modes={
 'Dirichlet':{k:BoundaryData('DIRICHLET',u) for k in tags},
 'Neumann_zero_mean':{k:BoundaryData('NEUMANN',flux) for k in tags},
 'mixed':{k:BoundaryData('DIRICHLET',u) if k==tags[0] else BoundaryData('NEUMANN',flux) for k in tags}}
 for mode,conditions in modes.items():
  fem=FEM(m,conditions=conditions,order=4);sol=fem.poisson(0.);target=truth.copy()
  if mode=='Neumann_zero_mean':target-=fem.mass_weights@target/fem.mass_weights.sum()
  error=float(abs(sol-target).max());require(error<2e-11,'affine D/N/mixed control failed')
  rows.append({'family':family,'mode':mode,'max_error':error,'residual':fem.last_residual,'Neumann_tags':sum(b.kind=='NEUMANN' for b in conditions.values()),'research_certification':'D0 only; nonzero D/N/mixed are solver controls, not certified smooth research policies'})
result={'passed':True,'source_revision':physics_identity(),'rows':rows};Path('reports/smooth_finalization/boundary-controls.json').write_text(json.dumps(result,indent=2)+'\n');print('passed24 boundary controls; maximum',max(r['max_error'] for r in rows))
