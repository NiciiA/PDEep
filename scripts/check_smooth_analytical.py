"""Independent analytical square T1 solutions through the new bulk time integrator.

Test-only analytic initial functions. No production field prior is modified.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
from pdeep.core import require
from pdeep.smooth.definition import geometry,random_stream,physics_identity
from pdeep.smooth.reference import Reference
from pdeep.wave_fidelity.reference import lagrange
from pdeep.geometric.quadrature import cells
from scipy.sparse.linalg import splu

class AnalyticalControl(Reference):
 def initial(self,fields):return np.column_stack([f(self.space.position)[self.space.free] for f in fields])

modes=np.array([(m,n) for m in (1,2,3) for n in (1,2,3)]);lam=np.pi**2*(modes*modes).sum(1);omega=.7*np.sqrt(lam);tau=.5*(lam[0]/lam)**.75
stream=np.random.default_rng(287);a=tau*stream.standard_normal(9);b=.4*omega*tau*stream.standard_normal(9)
def phi(x):return 2*np.sin(np.pi*x[:,0,None]*modes[:,0])*np.sin(np.pi*x[:,1,None]*modes[:,1])
u0=lambda x:phi(x)@a;v0=lambda x:phi(x)@b
r=AnalyticalControl('square',.04,4);mesh=geometry('square');x=mesh.sample(64,random_stream('analytical-control'));times=np.array([0.,.0073,.02,.05,.1,.237,.5,1.]);xx=np.tile(x,(len(times),1));tt=np.repeat(times,len(x));P=phi(xx)
truth_heat=np.sum(P*(np.exp(-.02*tt[:,None]*lam)*a),axis=1)
truth_wave=np.sum(P*(np.cos(tt[:,None]*omega)*a+np.sin(tt[:,None]*omega)*b/omega),axis=1)
truth_v=np.sum(P*(-np.sin(tt[:,None]*omega)*omega*a+np.cos(tt[:,None]*omega)*b),axis=1)
heat,_,dh=r.dynamic('heat',[u0],None,xx,tt,.002);wave,velocity,dw=r.dynamic('wave',[u0],[v0],xx,tt,.002)
rows=[]
for p,q,values,target in [('heat','u',heat[0],truth_heat),('wave','u',wave[0],truth_wave),('wave','v',velocity[0],truth_v)]:
 scale=float(np.sqrt(np.mean(target*target)));err=values-target;relative=float(np.sqrt(np.mean(err*err))/scale);require(relative<1e-4,'analytical square reference check failed')
 rows.append({'pde':p,'quantity':q,'physical_RMSE':float(np.sqrt(np.mean(err*err))),'relative_own_analytical_RMS':relative,'max_absolute_error':float(abs(err).max()),'time_RMSE':{str(t):float(np.sqrt(np.mean(err[i*len(x):(i+1)*len(x)]**2))) for i,t in enumerate(times)}})
s=r.space;quad=cells(s.mesh,8);basis=lagrange(quad.barycentric,4)[0];F=np.zeros(len(s.position));source=phi(quad.position)@(lam*a);np.add.at(F,s.dofs[quad.cell_id].ravel(),(quad.weight[:,None]*basis*source[:,None]).ravel());sol=splu(s.Ki).solve(F[s.free]);err=s.sample_operator(x)@sol-u0(x);relative=float(np.sqrt(np.mean(err*err))/np.sqrt(np.mean(u0(x)**2)));require(relative<1e-5,'analytical Poisson square check failed');rows.append({'pde':'poisson','quantity':'u','physical_RMSE':float(np.sqrt(np.mean(err*err))),'relative_own_analytical_RMS':relative,'max_absolute_error':float(abs(err).max())})
result={'passed':True,'source_revision':physics_identity(),'fixture':'unchanged nine-mode T1 formulas, independent NumPy implementation; CPU64 P4/.04,dt.002; no prior modification','rows':rows,'wave_energy_drift':dw['wave_relative_energy_drift'],'heat_max_mass_norm_increase':dh['heat_max_mass_norm_increase']}
Path('reports/smooth_finalization/analytical-controls.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
