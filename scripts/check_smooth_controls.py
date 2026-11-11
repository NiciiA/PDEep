"""Independent CPU64 controls; NEVER constructs or steps an optimizer."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,time
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import splu
from pdeep.core import require,digest
from pdeep.smooth.definition import *
from pdeep.smooth.reference import Reference,field_matrix
from pdeep.geometric.quadrature import cells,boundary
from pdeep.wave_fidelity.reference import Space,GaussWave,SpectralControl,lagrange


def main():
    start=time.perf_counter();out={};family='square';r=Reference(family,.16,2);s=r.space;f=[physical_fields(family,'wave','validation',i) for i in range(2)]
    x=geometry(family).sample(32,random_stream('independent-control-coordinates'));times=np.array([0.,.0073,.02,.237,.5,1.]);xx=np.tile(x,(len(times),1));tt=np.repeat(times,len(x))
    # Independently assembled speed-scaled Space + generalized eigensystem.
    control_space=Space(s.mesh,2,speed=.7);spectral=SpectralControl(control_space,control_space.initial(f[0]['u0']),control_space.initial(f[0]['v0']))
    values=[spectral.nodal(t) for t in times];Scontrol=control_space.sample_operator(x);exact_u=np.array([Scontrol@v[0] for v in values]);exact_v=np.array([Scontrol@v[1] for v in values])
    errors=[]
    for dt in (.02,.01,.005):
        u,v,d=r.dynamic('wave',[z['u0'] for z in f],[z['v0'] for z in f],xx,tt,dt)
        errors.append({'dt':dt,'u_RMSE':float(np.sqrt(np.mean((u[0].reshape(len(times),-1)-exact_u)**2))),'v_RMSE':float(np.sqrt(np.mean((v[0].reshape(len(times),-1)-exact_v)**2))),'energy_drift':d['wave_relative_energy_drift']})
    require(errors[-1]['u_RMSE']<errors[0]['u_RMSE']/20 and errors[-1]['v_RMSE']<errors[0]['v_RMSE']/8,'wave temporal convergence against full spectrum failed')
    u,v,_=r.dynamic('wave',[f[0]['u0']],[f[0]['v0']],xx,tt,.005)
    independently_coded=GaussWave(control_space,f[0]['u0'],f[0]['v0'])
    gu,gv=independently_coded.evaluate(x,times,dt=.005)
    difference=max(float(abs(u[0].reshape(len(times),-1)-gu).max()),float(abs(v[0].reshape(len(times),-1)-gv).max()))
    require(difference<1e-10,'grouped wave disagrees with established Gauss4 control')
    # Arbitrary-coordinate/time chunks and v = derivative of the C1 dense output.
    t0=np.full(len(x),.237123);eps=1e-6
    _,vel,_=r.dynamic('wave',[f[0]['u0']],[f[0]['v0']],x,t0,.005)
    plus,_,_=r.dynamic('wave',[f[0]['u0']],[f[0]['v0']],x,t0+eps,.005);minus,_,_=r.dynamic('wave',[f[0]['u0']],[f[0]['v0']],x,t0-eps,.005)
    derivative_error=float(abs((plus-minus)/(2*eps)-vel).max());require(derivative_error<2e-7,'wave dense velocity derivative failed')
    full=r.dynamic('wave',[z['u0'] for z in f],[z['v0'] for z in f],xx,tt,.005)
    chunks=[r.dynamic('wave',[z['u0'] for z in f],[z['v0'] for z in f],xx[a:a+17],tt[a:a+17],.005) for a in range(0,len(tt),17)]
    chunk_error=max(float(abs(full[k]-np.concatenate([v[k] for v in chunks],axis=1)).max()) for k in (0,1));require(chunk_error<5e-12,'reference query chunks changed trajectory')
    out['wave']={'full_generalized_eigen_control':spectral.diagnostics,'temporal_errors':errors,'independent_Gauss4_max_discrepancy':difference,'v_equals_du_dt_max_error':derivative_error,'query_chunk_max_error':chunk_error}
    # Heat exact semidiscrete exponential, independent full generalized eigenbasis.
    hf=physical_fields(family,'heat','validation',0)['u0'];initial=r.initial([hf])[:,0];lam,V=eigh(.02*s.Ki.toarray(),s.Mi.toarray());coefficient=V.T@(s.Mi@initial);S=s.sample_operator(x)
    exact=np.asarray(S@V)@(np.exp(-lam[:,None]*times)*coefficient[:,None]);heat=[]
    for dt in (.02,.01,.005):
        u,_,d=r.dynamic('heat',[hf],None,xx,tt,dt);heat.append({'dt':dt,'RMSE':float(np.sqrt(np.mean((u[0].reshape(len(times),-1).T-exact)**2))),'max_mass_norm_increase':d['heat_max_mass_norm_increase']})
    require(heat[-1]['RMSE']<heat[0]['RMSE']/10 and all(h['max_mass_norm_increase']<=1e-12 for h in heat),'CN convergence/dissipation failed')
    out['heat']={'temporal_errors':heat,'expected_order':'2 at fixed positive physical times (CN); C1 output derivative from semidiscrete heat equation'}
    # Poisson manufactured quartic, independently assembled source load.
    p4=Reference(family,.16,4);q=cells(p4.space.mesh,8);basis=lagrange(q.barycentric,4)[0];xq=q.position
    src=2*(xq[:,0]*(1-xq[:,0])+xq[:,1]*(1-xq[:,1]));load=np.zeros(len(p4.space.position));np.add.at(load,p4.space.dofs[q.cell_id].ravel(),(q.weight[:,None]*basis*src[:,None]).ravel());a=splu(p4.space.Ki).solve(load[p4.space.free]);y=p4.space.sample_operator(x)@a;truth=x[:,0]*(1-x[:,0])*x[:,1]*(1-x[:,1]);err=float(abs(y-truth).max());require(err<2e-12,'manufactured Poisson P4 control failed');out['poisson_quartic_max_error']=err
    geometry_rows=[]
    for family in GEOMETRIES:
        r=Reference(family,.08,3);b=boundary(r.space.mesh,4);m=r.space.mesh;epsilon=1e-7
        inward=m.features(b.position-epsilon*b.normal)['inside'];outward=m.features(b.position+epsilon*b.normal)['inside']
        require(inward.all() and not outward.any(),'outward/inner-hole normal sign failed')
        fs=[physical_fields(family,'poisson','validation' if family in SEEN else 'geometry_held_out',0)['f']]
        a,_=r.poisson(fs,x if family=='square' else m.sample(64,random_stream('source-quadrature',family)),8)
        pos=x if family=='square' else m.sample(64,random_stream('source-quadrature',family));b12,_=r.poisson(fs,pos,12)
        qerror=float(abs(a-b12).max());require(qerror<5e-7,'source quadrature8/12 difference too large')
        trace=max(float(np.max(abs(z(m.vertices[np.unique(m.boundary_edges)])))) for z in physical_fields(family,'wave','validation',0).values());require(trace<1e-12,'C3 trace compatibility failed')
        geometry_rows.append({'family':family,'normal_direction_failures':int((~inward).sum()+outward.sum()),'initial_trace_max':trace,'Poisson_source_quadrature8_vs12_max_absolute':qerror,'boundary_scope':'D0 certified; existing mixed/Neumann manufactured regression checked separately'})
    out['geometry_boundaries_and_source_quadrature']=geometry_rows;out['passed']=True;out['seconds']=time.perf_counter()-start;out['source_revision']=physics_identity()
    path=Path('reports/smooth_finalization/independent-controls.json');path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
