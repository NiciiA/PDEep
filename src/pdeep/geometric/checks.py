"""Independent scientific gates for geometry and CPU64 reference solvers."""
from pathlib import Path
import json,time
import numpy as np
from scipy.sparse.linalg import eigsh
from .domain import make_shape,mesh_shape,GeometryDomain
from .fem import FEM,BoundaryData
from .quadrature import cells,boundary
from ..core import require


def error(mesh,u,exact,*,t=0.,order=6):
    q=cells(mesh,order);pred=np.sum(u[mesh.cells[q.cell_id]]*q.barycentric,axis=1);truth=exact(q.position,t)
    e=pred-truth;l2=float(np.sqrt(q.weight@(e*e)));norm=float(np.sqrt(q.weight@(truth*truth)))
    return {'L2':l2,'relative_L2':l2/max(norm,1e-30),'max_pointwise':float(np.max(np.abs(e)))}


def run_foundation(output):
    start=time.perf_counter();geometries={};manufactured={};directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    families=('square','rectangle','lshape','disk','ellipse','hole','annulus','polygon')
    for family in families:
        shape=make_shape(family);rows=[];mms=[]
        for h in ((.08,.04,.02) if family=='annulus' else (.16,.08,.04)):
            t0=time.perf_counter();mesh=mesh_shape(shape,h);q=cells(mesh,4);qb=boundary(mesh,4)
            require(abs(q.weight.sum()-mesh.cell_areas.sum())<1e-12,'volume quadrature area fails')
            require(abs(qb.weight.sum()-mesh.boundary_lengths.sum())<1e-12,'boundary quadrature length fails')
            require(mesh.geometry_id==mesh_shape(shape,.2).geometry_id,'remeshing changed physical geometry identity')
            points=mesh.sample(41,np.random.default_rng(21));field=mesh.vertices[:,0]+.37*mesh.vertices[:,1]
            pe=float(np.max(np.abs(mesh.evaluate(field,points)-(points[:,0]+.37*points[:,1]))));require(pe<2e-14,'P1 affine interpolation fails')
            mesh.save(directory/(family+f'-{h}.npz'));require(GeometryDomain.load(directory/(family+f'-{h}.npz')).mesh_id==mesh.mesh_id,'geometry roundtrip failed')
            phi=lambda x,t:np.exp(x[:,0]+.3*x[:,1]);conditions={int(k):BoundaryData(value=phi) for k in set(mesh.boundary_tags)}
            f=FEM(mesh,conditions=conditions,order=6);u=f.poisson(lambda x,t:-1.09*phi(x,t))
            record={'target_h':h,**mesh.quality(),'poisson':error(mesh,u,phi),'residual':f.last_residual,'seconds':time.perf_counter()-t0,'mesh_id':mesh.mesh_id}
            # Same smooth manufactured solution on actual irregular domain; true
            # source and prescribed boundary are known verification data only.
            for pde,a in (('heat',.02),('wave',.7**2)):
                exact=(lambda x,t:phi(x,t)*np.exp(-t)) if pde=='heat' else (lambda x,t:phi(x,t)*np.cos(t))
                velocity=(lambda x,t:-phi(x,t)*np.exp(-t)) if pde=='heat' else (lambda x,t:-phi(x,t)*np.sin(t))
                source=lambda x,t,a=a,exact=exact:-(1+1.09*a)*exact(x,t)
                bc={int(k):BoundaryData(value=exact,velocity=velocity) for k in set(mesh.boundary_tags)}
                solver=FEM(mesh,coefficient=a,conditions=bc,order=6)
                solution=solver.dynamics(pde,phi,velocity=(lambda x,t:np.zeros(len(x))),source=source,dt=.002,final_time=.2)
                record[pde]=error(mesh,solution.nodal(.137),exact,t=.137)
                if pde=='wave':record['wave_velocity']=error(mesh,solution.nodal(.137,'v'),velocity,t=.137)
                record[pde+'_residual_max']=max(r['relative'] for r in solution.residuals)
            rows.append(record)
        require(rows[-1]['poisson']['L2']<rows[0]['poisson']['L2']/4,'elliptic refinement failed independent smooth manufactured solution')
        for pde in ('heat','wave','wave_velocity'):
            require(rows[-1][pde]['L2']<rows[0][pde]['L2'],'dynamic spatial refinement failed independent manufactured solution')
        for pde in ('poisson','heat','wave','wave_velocity'):
            for i in (1,2):rows[i][pde]['observed_rate']=float(np.log(rows[i-1][pde]['L2']/rows[i][pde]['L2'])/np.log(rows[i-1]['h_max']/rows[i]['h_max']))
        if family=='square':require(rows[-1]['poisson']['observed_rate']>1.7,'smooth square P1 L2 convergence below expected second order')
        geometries[family]=rows
        print('G1/G2 geometry and spatial MMS:',family,'passed',flush=True)
    # Independent temporal reference: exact evolution of a semi-discrete mode.
    # This isolates time error; the spatial checks above use continuous PDE MMS.
    temporal={};mesh=mesh_shape(make_shape('square'),.1)
    for pde,a in (('heat',.02),('wave',.49)):
        f=FEM(mesh,coefficient=a);I=f.free;K=f.K[I][:,I];M=f.M[I][:,I]
        lam,vec=eigsh(K,k=1,M=M,which='SM',v0=np.ones(len(I)));mode=np.zeros(len(mesh.vertices));mode[I]=vec[:,0]
        rows=[]
        for dt in (.08,.04,.02,.01):
            sol=f.dynamics(pde,mode,dt=dt);errs=[];verrs=[];energy=sol.energy()
            for t in (.137,.31,.79):
                exact=mode*np.exp(-lam[0]*t) if pde=='heat' else mode*np.cos(np.sqrt(lam[0])*t)
                e=sol.nodal(t)-exact;errs.append(float(np.sqrt(e@f.M@e)))
                if pde=='wave':
                    v=-np.sqrt(lam[0])*mode*np.sin(np.sqrt(lam[0])*t);e=sol.nodal(t,'v')-v;verrs.append(float(np.sqrt(e@f.M@e)))
                    delta=1e-5
                    # Hermite output is C1, not generally C2 at time nodes.
                    # Use an independent 3-point one-sided derivative at knots,
                    # and a central derivative strictly inside each interval.
                    knot_distance=min(abs(t-z) for z in sol.times)
                    ad=(-3*sol.nodal(t)+4*sol.nodal(t+delta)-sol.nodal(t+2*delta))/(2*delta) if knot_distance<2*delta else (sol.nodal(t+delta)-sol.nodal(t-delta))/(2*delta)
                    require(np.max(np.abs(ad-sol.nodal(t,'v')))<1e-8,'Hermite physical velocity inconsistent with du/dt')
            if pde=='heat':require(np.all(np.diff(np.einsum('ti,ij,tj->t',sol.U,f.M.toarray(),sol.U))<1e-11),'heat mass norm does not dissipate')
            else:require(np.max(np.abs(energy-energy[0]))/(1+energy[0])<1e-10,'wave midpoint energy fails')
            rows.append({'dt':sol.dt,'L2_time_error':max(errs),'velocity_L2_time_error':max(verrs,default=0.),'energy_relative_drift':float(np.max(np.abs(energy-energy[0]))/(1+energy[0]))})
        for i in range(1,len(rows)):
            rows[i]['observed_time_rate']=float(np.log(rows[i-1]['L2_time_error']/rows[i]['L2_time_error'])/np.log(rows[i-1]['dt']/rows[i]['dt']))
        require(rows[-1]['observed_time_rate']>1.8,'second-order temporal convergence fails')
        temporal[pde]=rows
    # Mixed flux signs and pure-Neumann compatibility/gauge, independent affine/quadratic solutions.
    mesh=mesh_shape(make_shape('square'),.08)
    exact=lambda x,t:x[:,0]+2*x[:,1]
    mixed={k:BoundaryData('DIRICHLET',exact) if k==4 else BoundaryData('NEUMANN',lambda x,t,n:n[:,0]+2*n[:,1]) for k in range(1,5)}
    f=FEM(mesh,conditions=mixed);u=f.poisson();mixed_error=error(mesh,u,exact)['max_pointwise'];require(mixed_error<1e-11,'mixed Neumann sign/Dirichlet elimination fails')
    neumann={k:BoundaryData('NEUMANN',lambda x,t,n:2*np.sum(x*n,axis=1)) for k in range(1,5)}
    f=FEM(mesh,conditions=neumann);u=f.poisson(-4.);neumann_error=error(mesh,u,lambda x,t:np.sum(x*x,axis=1)-2/3)['L2']
    require(abs(f.mass_weights@u)<1e-11 and neumann_error<.01,'Neumann gauge/manufactured solution fails')
    try:f.poisson(1.)
    except ValueError:pass
    else:raise AssertionError('incompatible pure-Neumann problem was accepted')
    # Existing T1 nine/18-mode physics remains independently callable and is
    # the continuous reference here, not a discrete FEM-generated target.
    import torch
    from ..reference import evaluate,prior_scales
    legacy={}
    for pde in ('poisson','heat','wave'):
        xi=prior_scales(pde)*torch.tensor(np.random.default_rng(714).normal(size=18 if pde=='wave' else 9),dtype=torch.float64)
        def analytic(x,t=0.,quantity='u',pde=pde,xi=xi):
            position=torch.tensor(x,dtype=torch.float64)
            time=None if pde=='poisson' or quantity in {'f','u0','v0'} else torch.full((len(x),1),t,dtype=torch.float64)
            return evaluate(pde,xi,position,time,quantity=quantity).numpy()[:,0]
        records=[]
        for h in (.16,.08,.04):
            mesh=mesh_shape(make_shape('square'),h);f=FEM(mesh,coefficient=1. if pde=='poisson' else .02 if pde=='heat' else .49,order=6)
            if pde=='poisson':u=f.poisson(lambda x,t:analytic(x,t,'f'));v=None
            else:
                sol=f.dynamics(pde,lambda x,t:analytic(x,0,'u0'),velocity=(lambda x,t:analytic(x,0,'v0')) if pde=='wave' else None,dt=.001)
                u=sol.nodal(.237);v=sol.nodal(.237,'v') if pde=='wave' else None
            row={'h':h,'vertices':len(mesh.vertices),'u':error(mesh,u,analytic,t=.237)}
            if v is not None:row['v']=error(mesh,v,lambda x,t:analytic(x,t,'v'),t=.237)
            records.append(row)
        for quantity in ('u','v') if pde=='wave' else ('u',):
            require(records[-1][quantity]['L2']<records[0][quantity]['L2'] and records[-1][quantity]['relative_L2']<.12,'numerical FEM does not approach independent T1 analytic fields')
        legacy[pde]=records
    result={'passed':True,'gate':'G1+G2','geometries':geometries,'temporal':temporal,'T1_analytic_comparison':legacy,'mixed_affine_max_error':mixed_error,
            'Neumann_quadratic_L2_error':neumann_error,'CPU_precision':'float64','seconds':time.perf_counter()-start}
    (directory/'foundation.json').write_text(json.dumps(result,indent=2)+'\n');return result
