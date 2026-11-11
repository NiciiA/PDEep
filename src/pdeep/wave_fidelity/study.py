"""Deterministic no-training G3 audit, independently isolated error sources."""
from pathlib import Path
import gc,json,time
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
from ..core import digest,require
from ..geometric.data import domain,sample_field,rng,VERSION as PRIOR_VERSION,Normalizers
from ..geometric.fem import FEM
from ..geometric.domain import make_shape,mesh_shape
from .reference import Space,ContinuousWave,SpectralControl,GaussWave,refine
from .fields import pair,VERSION as SMOOTH_VERSION

TIMES=np.array([0.,.0073,.02,.05,.1,.237,.5,1.])
FAMILIES=(('square',1.,'validation',0),('rectangle',1.2,'validation',1),('lshape',1.,'validation',3),
          ('disk',1.,'validation',4),('ellipse',1.8,'geometry_held_out',3),('hole',1.,'geometry_held_out',0),
          ('annulus',1.,'geometry_held_out',1),('polygon',1.,'geometry_held_out',2))
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/wave_fidelity_g3'


def write(name,value):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/name).write_text(json.dumps(value,indent=2)+'\n')


def frozen_v1(mesh,split,index,length=.2):
    cid=digest((PRIOR_VERSION,split,'wave',index,mesh.geometry_id))
    return tuple(sample_field(mesh,rng(cid,'physical_field',q),zero_trace=True,length=length) for q in ('u0','v0'))


def positions(mesh):return mesh.sample(256,np.random.default_rng(20261111))


def errors(U,V,Uref,Vref,scales):
    return {q:[{'time':float(t),'physical_RMSE':float(np.sqrt(np.mean((a-b)**2))),
                'standardized_RMSE':float(np.sqrt(np.mean((a-b)**2))/scales[q]),
                'physical_max_error':float(np.max(abs(a-b))),
                'standardized_max_error':float(np.max(abs(a-b))/scales[q])} for t,a,b in zip(TIMES,values,truth)]
            for q,values,truth in (('u',U,Uref),('v',V,Vref))}


def temporal():
    m=domain('square',1.,.08);initial,velocity=frozen_v1(m,'validation',0);space=Space(m,1);u0=space.initial(initial);v0=space.initial(velocity);control=SpectralControl(space,u0,v0)
    exact=[control.nodal(float(t)) for t in TIMES];c=ContinuousWave(space,initial,velocity);cu,cv=c.evaluate(space.position[space.free],TIMES,return_nodal=True)
    direct_max=max(np.max(abs(cu[i]-u)) for i,(u,v) in enumerate(exact));direct_v=max(np.max(abs(cv[i]-v)) for i,(u,v) in enumerate(exact))
    n=len(space.free);A=np.block([[np.zeros((n,n)),np.eye(n)],[-np.linalg.solve(space.Mi.toarray(),space.Ki.toarray()),np.zeros((n,n))]])
    independent=expm(.237*A)@np.r_[u0,v0];ue,ve=control.nodal(.237)
    rows=[];fem=FEM(m,coefficient=.49)
    for dt in (.008,.004,.002,.001,.0005):
        sol=fem.dynamics('wave',initial,velocity=velocity,dt=dt);energy=sol.energy();eu=[];ev=[];initial_energy=space.energy(u0,v0)
        for t,(u,v) in zip(TIMES,exact):
            a=sol.nodal(float(t))[space.free];b=sol.nodal(float(t),'v')[space.free]
            eu.append(space.norm(a-u));ev.append(space.norm(b-v))
        row={'dt':sol.dt,'u_L2_error_by_time':eu,'v_L2_error_by_time':ev,'max_u_L2_error':max(eu),'max_v_L2_error':max(ev),
             'relative_grid_energy_drift':float(max(abs(energy-energy[0]))/energy[0]),
             'off_grid_relative_energy_error':max(abs(space.energy(sol.nodal(float(t))[space.free],sol.nodal(float(t),'v')[space.free])-initial_energy)/initial_energy for t in TIMES)}
        if rows:
            row['u_rate']=float(np.log(rows[-1]['max_u_L2_error']/row['max_u_L2_error'])/np.log(rows[-1]['dt']/sol.dt));row['v_rate']=float(np.log(rows[-1]['max_v_L2_error']/row['max_v_L2_error'])/np.log(rows[-1]['dt']/sol.dt))
        rows.append(row)
    energy=.5*(control.lam*control.a**2+control.b**2);spectrum=[]
    for cutoff in (5.,10.,15.,20.,30.,40.):spectrum.append({'omega_cutoff':cutoff,'initial_energy_fraction_above':float(energy[control.omega>cutoff].sum()/energy.sum())})
    amplification=[{'omega_dt':z,'amplitude':float(abs((1+.5j*z)/(1-.5j*z))),'phase':float(2*np.arctan(z/2)),'phase_lag_per_step':float(z-2*np.arctan(z/2))} for z in (.01,.1,.5,1.,2.,5.)]
    result={'times':TIMES.tolist(),'eigen_diagnostics':control.diagnostics,'Chebyshev_vs_eigen_max_u':float(direct_max),'Chebyshev_vs_eigen_max_v':float(direct_v),
            'expm_vs_eigen_max':float(np.max(abs(independent-np.r_[ue,ve]))),'Chebyshev_scalar_diagnostics':c.last_diagnostics,'temporal':rows,
            'spectrum':spectrum,'amplification':amplification,'scheme':'Cayley implicit midpoint, theta=2 atan(omega dt/2); modulus1, second-order phase error'}
    require(direct_max<1e-10 and direct_v<1e-9 and result['expm_vs_eigen_max']<1e-9,'independent semidiscrete controls disagree')
    require(rows[-1]['u_rate']>1.9 and rows[-1]['v_rate']>1.9,'midpoint temporal order failed')
    write('temporal.json',result);return result


def analytical():
    import torch
    from ..reference import evaluate,prior_scales
    xi=prior_scales('wave')*torch.tensor(np.random.default_rng(714).normal(size=18),dtype=torch.float64)
    def true(x,t=0.,quantity='u'):
        return evaluate('wave',xi,torch.tensor(x,dtype=torch.float64),torch.full((len(x),1),float(t),dtype=torch.float64),quantity=quantity).numpy()[:,0]
    initial=lambda x,t=0.:true(x,0.,'u');velocity=lambda x,t=0.:true(x,0.,'v')
    x=np.random.default_rng(200).uniform(.05,.95,(256,2));U=np.array([true(x,t,'u') for t in TIMES]);V=np.array([true(x,t,'v') for t in TIMES]);scales={'u':float(np.sqrt(np.mean(U*U))),'v':float(np.sqrt(np.mean(V*V)))}
    rows=[]
    for h,degree in ((.04,1),(.02,1),(.01,1),(.04,2),(.02,2),(.04,4)):
        start=time.perf_counter();mesh=domain('square',1.,h);s=Space(mesh,degree);w=ContinuousWave(s,initial,velocity);u,v=w.evaluate(x,TIMES)
        rows.append({'h':h,'degree':degree,'free_dofs':len(s.free),'actual_h_max':mesh.quality()['h_max'],'seconds':time.perf_counter()-start,'errors_against_analytical':errors(u,v,U,V,scales),
                     'time_approximation':w.last_diagnostics,'mesh_id':mesh.mesh_id})
        print('T1 analytical',h,degree,'max normalized v',max(z['standardized_RMSE'] for z in rows[-1]['errors_against_analytical']['v']),flush=True)
        del s,w;gc.collect()
    result={'times':TIMES.tolist(),'coordinate_seed':200,'coefficient_seed':714,'scale_definition':'space/time RMS of this analytic fixture (not a dataset noise calibrator)','scales':scales,'rows':rows}
    write('analytical-square.json',result);return result


def original():
    saved=ROOT/'reports/geometry_phase/normalizers.json'
    if saved.exists():n=Normalizers.fromdict(json.loads(saved.read_text()))
    else:
        # Clean checkouts have no gitignored Phase G outputs. Reproduce its
        # declared training-only calibration, never calibrate on this suite.
        from ..geometric.data import calibrate
        n=calibrate(5,64)
        print('Recreated original declared TRAIN-only calibration5/64; verify identity against your archived normalizers.',flush=True)
    write('original-normalizers.json',n.asdict());rows=[]
    for family,aspect,split,index in FAMILIES:
        path=OUT/f'original-{family}.json'
        if path.exists():rows.append(json.loads(path.read_text()));continue
        canonical=domain(family,aspect,.08);initial,velocity=frozen_v1(canonical,split,index);x=positions(canonical);values=[];t0=time.perf_counter()
        exact_initial_u=initial(x);exact_initial_v=velocity(x)
        for h in (.04,.02,.01):
            mesh=domain(family,aspect,h);s=Space(mesh,1);w=ContinuousWave(s,initial,velocity);u,v=w.evaluate(x,TIMES)
            values.append((h,u,v,{'h':h,'free_dofs':len(s.free),'actual_h_max':mesh.quality()['h_max'],'mesh_id':mesh.mesh_id,'initial_energy':s.energy(w.u0,w.v0),
                                   'time_approximation':w.last_diagnostics,'initial_u_RMSE':float(np.sqrt(np.mean((u[0]-exact_initial_u)**2))),'initial_v_RMSE':float(np.sqrt(np.mean((v[0]-exact_initial_v)**2)))}))
            del s,w;gc.collect()
        scales={q:n.scale('wave',q) for q in ('u','v')};row={'family':family,'aspect':aspect,'prior':initial.prior,'field_digests':[digest(z.nodal.tolist()) for z in (initial,velocity)],'times':TIMES.tolist(),
             'coordinates':x.tolist(),'error_definition':'mesh-to-mesh; finest h=.01 is NOT certified exact','reference_mesh':'h=.01 P1, direct semidiscrete propagation',
             'rows':[{**info,'differences_to_h01':errors(u,v,values[-1][1],values[-1][2],scales)} for h,u,v,info in values[:-1]],
             'finest':values[-1][3],'seconds':time.perf_counter()-t0}
        write(path.name,row);rows.append(row);print('Original prior',family,'h.02->.01 max std v',max(z['standardized_RMSE'] for z in row['rows'][-1]['differences_to_h01']['v']),flush=True)
    write('original-prior.json',{'scales':{q:n.scale('wave',q) for q in ('u','v')},'families':rows,'release':'BLOCKED; field prior unchanged'});return rows


def correlation():
    mesh=domain('square',1.,.08);space=Space(mesh,1);rows=[]
    for length in (.1,.2,.35,.5):
        fields=tuple(sample_field(mesh,np.random.default_rng(k),zero_trace=True,length=length) for k in (711,712));u0=space.initial(fields[0]);v0=space.initial(fields[1]);control=SpectralControl(space,u0,v0)
        E=.5*(control.lam*control.a**2+control.b**2)
        rows.append({'length':length,'same_white_noise_seeds':[711,712],'total_energy':float(E.sum()),'above_omega15_fraction':float(E[control.omega>15].sum()/E.sum()),
                     'above_omega30_fraction':float(E[control.omega>30].sum()/E.sum()),'eigen_omega':control.omega.tolist(),'energy_by_eigenmode':E.tolist()})
    write('correlation-length.json',{'rows':rows,'semantics':'Explicit prior-parameter diagnostic, never substituted into canonical benchmark'});return rows


def candidate_calibration():
    path=OUT/'candidate-normalizers.json'
    if path.exists():return json.loads(path.read_text())
    rows=[]
    for family,aspect,split,index in FAMILIES[:4]: # training catalog square/rect/L/disk only
        canonical=domain(family,aspect,.08);fields=pair(canonical,('train-calibration',0),relative_radius=.18)
        s=Space(domain(family,aspect,.02),4);w=ContinuousWave(s,*fields);x=canonical.sample(256,np.random.default_rng(501));z,weights=np.polynomial.legendre.leggauss(8);u,v=w.evaluate(x,(z+1)/2)
        rows.append({'family':family,'field_ids':[f.identity for f in fields],'mean_square_u':float(weights@np.mean(u*u,axis=1)/2),'mean_square_v':float(weights@np.mean(v*v,axis=1)/2)})
        del s,w;gc.collect();print('Candidate training-only calibration',family,flush=True)
    result={'version':SMOOTH_VERSION,'relative_radius':.18,'split':'train-calibration; no validation/test','reference':'P4 h=.02, direct semidiscrete time',
            'area_uniform_points_per_case':256,'time_Gauss_order':8,'rows':rows,'scales':{q:float(np.sqrt(np.mean([r['mean_square_'+q] for r in rows]))) for q in ('u','v')}}
    write(path.name,result);return result


def candidate():
    calibration=candidate_calibration();scales=calibration['scales'];rows=[]
    for family,aspect,split,index in FAMILIES:
        path=OUT/f'candidate-{family}.json'
        if path.exists():rows.append(json.loads(path.read_text()));continue
        canonical=domain(family,aspect,.08);fields=pair(canonical,('validation',0),relative_radius=.18);x=positions(canonical);values=[];start=time.perf_counter()
        for h,degree in ((.04,2),(.02,2),(.01,2),(.04,4),(.02,4)):
            u,v,info=cached_candidate(family,aspect,fields,x,h,degree)
            values.append((h,degree,u,v,info))
            print('Candidate',family,h,'P'+str(degree),'done',flush=True)
        reference=values[-1];row={'family':family,'fields':[f.specification for f in fields],'times':TIMES.tolist(),'coordinates':x.tolist(),'reference':'P4 h=.02 (refinement comparator, not exact truth)',
              'scales':scales,'rows':[{**info,'differences_to_P4_h02':errors(u,v,reference[2],reference[3],scales)} for h,d,u,v,info in values[:-1]],'seconds':time.perf_counter()-start}
        # P3 h=.01 is a distinct discretization, and finer P4 is checked on the
        # worst families separately; do not turn same-mesh p agreement into proof.
        u,v,info=cached_candidate(family,aspect,fields,x,.01,3,method='gauss',dt=.004)
        uf,vf,fine_info=cached_candidate(family,aspect,fields,x,.01,3,method='gauss',dt=.002)
        row['fine_temporal_halving']=errors(u,v,uf,vf,scales);row['fine_reference_diagnostics']=fine_info
        if max(e['standardized_RMSE'] for q in ('u','v') for e in row['fine_temporal_halving'][q])>=.0002:
            u,v=uf,vf;uf,vf,fine_info=cached_candidate(family,aspect,fields,x,.01,3,method='gauss',dt=.001)
            row['fine_temporal_halving']=errors(u,v,uf,vf,scales);row['fine_reference_diagnostics']=fine_info
        row['P4_h02_vs_P3_h01']=errors(reference[2],reference[3],uf,vf,scales)
        row['necessary_RMS_gate']=max(e['standardized_RMSE'] for q in ('u','v') for e in row['P4_h02_vs_P3_h01'][q])<.002
        row['fine_temporal_gate']=max(e['standardized_RMSE'] for q in ('u','v') for e in row['fine_temporal_halving'][q])<.0002
        write(path.name,row);rows.append(row)
    write('candidate-prior.json',{'calibration':calibration,'families':rows,'criterion':'RMS <.002 training RMS; auxiliary pointwise <.01, same-field finer-reference checks required','all_necessary_RMS_gates_pass':all(r['necessary_RMS_gate'] for r in rows),
            'benchmark_switched':False,'original_V1_gate':'BLOCKED'});return rows


def cached_candidate(family,aspect,fields,x,h,degree,*,method='cheb',dt=None):
    """Per-configuration numerical checkpoint: no failed late gate loses a suite."""
    import hashlib
    from . import reference as implementation
    source=hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest()
    identity=digest((source,family,aspect,[f.identity for f in fields],x.tolist(),TIMES.tolist(),h,degree,method,dt))
    directory=OUT/'candidate-values';directory.mkdir(parents=True,exist_ok=True);path=directory/(identity+'.npz')
    if path.exists():
        data=np.load(path,allow_pickle=False);return data['u'],data['v'],json.loads(str(data['metadata']))
    # Reuse pre-optimization Chebyshev measurements only after proving every
    # class/function involved is byte-for-byte identical as Python AST. The
    # optimized Gauss implementation must never relabel old measurements.
    previous=OUT/'reference-before-fast-gauss.py'
    if method=='cheb' and previous.exists():
        import ast
        def relevant(path):
            return [ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in {'Space','lagrange','indices','coefficients','ContinuousWave'}]
        if relevant(previous)==relevant(Path(implementation.__file__)):
            old_source=hashlib.sha256(previous.read_bytes()).hexdigest();old_identity=digest((old_source,family,aspect,[f.identity for f in fields],x.tolist(),TIMES.tolist(),h,degree,method,dt))
            old_path=directory/(old_identity+'.npz')
            if old_path.exists():
                data=np.load(old_path,allow_pickle=False);return data['u'],data['v'],json.loads(str(data['metadata']))
    start=time.perf_counter();space=Space(domain(family,aspect,h),degree)
    if method=='cheb':w=ContinuousWave(space,*fields);u,v=w.evaluate(x,TIMES)
    else:w=GaussWave(space,*fields);u,v=w.evaluate(x,TIMES,dt=dt)
    info={'identity':identity,'source':source,'family':family,'h':h,'degree':degree,'dofs':len(space.free),'mesh_id':space.mesh.mesh_id,
          'actual_h_max':space.mesh.quality()['h_max'],'method':method,'seconds':time.perf_counter()-start,'time_approximation':w.last_diagnostics}
    np.savez_compressed(path,u=u,v=v,metadata=json.dumps(info));del space,w;gc.collect()
    print('Reference cached:',family,degree,h,method,dt,'seconds',info['seconds'],flush=True);return u,v,info
