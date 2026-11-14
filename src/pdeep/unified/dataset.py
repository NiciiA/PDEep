"""Incremental, explicit-index dataset preparation and quarantine; CPU64 only."""
from dataclasses import asdict
from pathlib import Path
import json, time
import numpy as np
import torch
from ..core import digest, require
from ..artifact_io import atomic_json, read_payload, write_artifact, read_manifest
from ..smooth.data import Case
from ..smooth.definition import POLICY, physical_fields, geometry, random_stream, split_manifest
from ..smooth.reference import Reference, reference
from ..smooth.certification import authorize, normalizers
from .config import DatasetConfig, SCHEMA, TIMES, case_index
from .reference import LinearCache, CACHE_SCHEMA


def case_from_key(key,conditioning_h=.12):
    p,g,s,i=key;return Case(g,p,s,i,conditioning_h)


def inventory(config):
    """No numerical reference construction. Hidden metadata never enters Problem."""
    rows=[];ids=set();latents=set();fields=set()
    for split in ('train','validation','test'):
        for key in config.keys(split):
            c=case_from_key(key,config.conditioning_h);f=c.fields
            latent=digest({k:v.coefficients.tolist() for k,v in f.items()});field=digest({k:v.identity for k,v in f.items()})
            require(c.case_id not in ids and latent not in latents and field not in fields,'duplicate physical identity/latent/field across dataset')
            ids.add(c.case_id);latents.add(latent);fields.add(field)
            rows.append({'key':key,'case_id':c.case_id,'latent_digest':latent,'field_digest':field,
                         'geometry_id':c.problem.mesh.geometry_id,'observation_seed':digest((c.case_id,'fixed-evaluation-observations',config.seed)),
                         'evaluation_seed':digest((c.case_id,'independent-evaluation',config.seed))})
    return rows


def _design(c,n):
    x=geometry(c.family,.08).sample(n,random_stream('unified-reference-QC',c.case_id))
    times=(0.,) if c.pde=='poisson' else TIMES
    qset=(0,1) if c.pde=='wave' else (0,)
    xx=np.tile(x,(len(times)*len(qset),1));tt=np.tile(np.repeat(times,n),len(qset));qq=np.repeat(qset,n*len(times))
    # Additional arbitrary non-grid times; never validation/test calibration.
    if c.pde!='poisson':
        xx=np.vstack((xx,geometry(c.family,.08).sample(n,random_stream('QC-arbitrary-x',c.case_id))))
        tt=np.r_[tt,random_stream('QC-arbitrary-t',c.case_id).uniform(0,1,n)];qq=np.r_[qq,np.arange(n)%len(qset)]
    return xx,tt,qq


def _full(c,r,x,t,q,dt):
    f=c.fields
    if c.pde=='poisson':u,d=r.poisson([f['f']],x,8);return u[0],d
    u,v,d=r.dynamic(c.pde,[f['u0']],[f['v0']] if c.pde=='wave' else None,x,t,dt)
    return (np.where(q==1,v[0],u[0]) if v is not None else u[0]),d


def _errors(a,b,q,t,c,norms):
    rows=[]
    for k in (0,1) if c.pde=='wave' else (0,):
        for time in [None,*(() if c.pde=='poisson' else TIMES)]:
            mask=(q==k) if time is None else (q==k)&(t==time)
            if mask.any():
                d=(a[mask]-b[mask])/norms.scale(c.pde,k)
                rows.append({'quantity':'v' if k else 'u','time':time,'standardized_RMSE':float(np.sqrt(np.mean(d*d))),
                             'standardized_max':float(np.max(abs(d)))})
    return rows


def quality_check(c,cache,norms,config,*,finer=False):
    x,t,q=_design(c,config.qc_points);r=reference(c.family,.02,4)
    full,diagnostics=_full(c,r,x,t,q,.002);cached,bound=cache.evaluate(c,x,t,q,return_bound=True)
    scales=np.array([norms.scale(c.pde,int(k)) for k in q]);errors=_errors(cached,full,q,t,c,norms)
    checks={'finite':bool(np.isfinite(full).all() and np.isfinite(cached).all()),
            'cache_agreement':max(e['standardized_RMSE'] for e in errors)<=config.cache_tolerance,
            'compression_bound':float(np.max(bound/scales))<=config.cache_tolerance,
            'solve_residual':diagnostics.get('max_relative_solve_residual',diagnostics.get('max_relative_residual',1.))<1e-10}
    # Reference.initial checks every Dirichlet nodal initial value. Also record exact
    # nodal trace and initial injection (no comparison of unlike interpolation spaces).
    if c.pde!='poisson':
        initial=r.initial([c.fields['u0']]);expected=c.fields['u0'](r.space.position[r.space.free])
        checks['initial_nodal_injection']=bool(np.max(abs(initial[:,0]-expected))<1e-12)
        checks['initial_trace']=all(np.max(abs(f(r.space.position[r.space.dirichlet])),initial=0.)<1e-10 for f in c.fields.values())
        if c.pde=='wave':
            vel=r.initial([c.fields['v0']]);checks['initial_velocity_injection']=bool(np.max(abs(vel[:,0]-c.fields['v0'](r.space.position[r.space.free])))<1e-12)
            checks['discrete_energy']=diagnostics['wave_relative_energy_drift']<1e-8
            xx=x[:min(8,len(x))];tt=np.linspace(.0073,.9473,len(xx));h=1e-6
            fd=(cache.evaluate(c,xx,tt+h,np.zeros(len(xx),int))-cache.evaluate(c,xx,tt-h,np.zeros(len(xx),int)))/(2*h)
            vv=cache.evaluate(c,xx,tt,np.ones(len(xx),int));kin=float(np.max(abs(fd-vv)))/norms.scale('wave','v')
            diagnostics['kinematic_FD_max_standardized_error']=kin;checks['kinematic_consistency']=kin<1e-5
        else:checks['mass_norm_nonincrease']=diagnostics['heat_max_mass_norm_increase']<1e-10
    # Tagged homogeneous trace: independent boundary query operator against the solver.
    xb=r.space.mesh.vertices[np.unique(r.space.mesh.boundary_edges)][::max(1,len(r.space.mesh.boundary_edges)//32)]
    boundary,_=_full(c,r,xb,np.full(len(xb),.237),np.zeros(len(xb),int),.002)
    checks['boundary_trace']=float(np.max(abs(boundary),initial=0.))<1e-10
    comparisons={}
    if finer:
        fine,fd=_full(c,Reference(c.family,.01,3),x,t,q,.001)
        comparisons['independent_P3_h001_dt0001']={'errors':_errors(full,fine,q,t,c,norms),'solver':fd,
                                                      'label':'mesh/method difference; not exact continuum error'}
        if c.pde!='poisson':
            temporal,td=_full(c,r,x,t,q,.001)
            comparisons['P4_h002_dt0001']={'errors':_errors(full,temporal,q,t,c,norms),'solver':td}
        checks['finer_spatial']=max(e['standardized_RMSE'] for e in comparisons['independent_P3_h001_dt0001']['errors'])<=config.reference_tolerance
        if c.pde!='poisson':checks['temporal']=max(e['standardized_RMSE'] for e in comparisons['P4_h002_dt0001']['errors'])<=.0002
    return {'passed':all(checks.values()),'checks':checks,'solver':diagnostics,'cache_errors':errors,
            'max_standardized_compression_bound':float(np.max(bound/scales)), 'comparisons':comparisons,
            'reference_policy':POLICY,'scope':'per-case invariants + compression/full-solver comparison; finer checks stratified, finite evidence'}


def quality_check_full(c,full,norms,config,*,finer=False):
    """Independent algebraic step/energy checks of the original FULL trajectory.

    No second solve is needed per case: Gauss4's complex delta is reconstructed
    from endpoint differences, then checked against M/K and its original RHS.
    Independent finer propagation is still required on the stratified subset.
    """
    from .reference import CACHE_SCHEMA
    from ..smooth.definition import COEFFICIENTS
    x,t,q=_design(c,config.qc_points);y=full.evaluate(x,t,q);r=full.reference;s=r.space;f=c.fields
    checks={'finite':bool(np.isfinite(y).all())};diagnostics={'precision':'CPU float64','point_labels':'direct original full P4 sample/C1 Hermite; no compression or finite-bank recycling'}
    if c.pde=='poisson':
        F=r.load([f['f']],8)[:,0];rr=np.linalg.norm(s.Ki@full.u[0]-F)/(1+np.linalg.norm(F));checks['solve_residual']=rr<1e-10
    else:
        checks['initial_injection']=bool(np.max(abs(full.u[0]-f['u0'](s.position[s.free])))<1e-12)
        checks['initial_trace']=all(np.max(abs(ff(s.position[s.dirichlet])),initial=0.)<1e-10 for ff in f.values())
        K=COEFFICIENTS[c.pde]*s.Ki;h=full.dt;steps=np.unique(np.linspace(0,len(full.u)-2,51,dtype=int));energies=[];rr=0.
        for k in steps:
            ul,ur=full.u[k],full.u[k+1]
            if c.pde=='wave':
                vl,vr=full.v[k],full.v[k+1];gamma=(1+1j*np.sqrt(3))/24;beta=.25+1j/(4*np.sqrt(3))
                zi=(ur-ul)/(4*np.sqrt(3));z=np.sqrt(3)*zi-h*(vr-vl)/12+1j*zi
                rhs=beta*h*(s.Mi@vl)-gamma*h*h*(K@ul);res=s.Mi@z+gamma*h*h*(K@z)-rhs
                energies.append(float(.5*(ul@(K@ul)+vl@(s.Mi@vl))))
            else:
                rhs=s.Mi@ul-h/2*(K@ul);res=s.Mi@ur+h/2*(K@ur)-rhs
                energies.append(float(ul@(s.Mi@ul)))
            rr=max(rr,float(np.linalg.norm(res)/(1+np.linalg.norm(rhs))))
        checks['step_residual']=rr<1e-10
        final_u=full.u[-1]
        energies.append(float(.5*(final_u@(K@final_u)+full.v[-1]@(s.Mi@full.v[-1]))) if c.pde=='wave' else float(final_u@(s.Mi@final_u)))
        diagnostics['energy_grid_states_checked']=len(energies)
        diagnostics['energy_includes_t0_t1']=True
        if c.pde=='wave':
            checks['initial_velocity']=bool(np.max(abs(full.v[0]-f['v0'](s.position[s.free])))<1e-12)
            drift=float(np.max(abs(np.array(energies)-energies[0]))/max(energies[0],1e-30));checks['energy']=drift<1e-8
            xx=x[:8];tt=np.linspace(.0073,.9473,len(xx));hh=1e-6
            fd=(full.evaluate(xx,tt+hh,np.zeros(len(xx),int))-full.evaluate(xx,tt-hh,np.zeros(len(xx),int)))/(2*hh)
            kin=float(np.max(abs(fd-full.evaluate(xx,tt,np.ones(len(xx),int)))))/norms.scale('wave','v');checks['kinematic']=kin<1e-5
            diagnostics.update(wave_relative_energy_drift=drift,kinematic_FD_max_standardized_error=kin)
        else:
            increase=float(np.max(np.diff(energies),initial=0.));checks['mass_norm_nonincrease']=increase<1e-10;diagnostics['heat_max_mass_norm_increase']=increase
    diagnostics['max_relative_step_residual']=float(rr)
    xb=s.mesh.vertices[np.unique(s.mesh.boundary_edges)][::max(1,len(s.mesh.boundary_edges)//32)]
    trace=full.evaluate(xb,np.full(len(xb),.237),np.zeros(len(xb),int));checks['boundary']=float(np.max(abs(trace),initial=0.))<1e-10
    comparisons={}
    if finer:
        fine,fd=_full(c,Reference(c.family,.01,3),x,t,q,.001);errors=_errors(y,fine,q,t,c,norms)
        comparisons['independent_P3_h001_dt0001']={'errors':errors,'solver':fd,'label':'mesh/method difference, not exact continuum error'}
        checks['finer_spatial']=max(row['standardized_RMSE'] for row in errors)<=config.reference_tolerance
        if c.pde!='poisson':
            temporal,td=_full(c,r,x,t,q,.001);errors=_errors(y,temporal,q,t,c,norms)
            comparisons['P4_h002_dt0001']={'errors':errors,'solver':td};checks['temporal']=max(row['standardized_RMSE'] for row in errors)<=.0002
    return {'passed':all(checks.values()),'checks':checks,'solver':diagnostics,'comparisons':comparisons,'reference_policy':POLICY,
            'scope':'per-case FULL CPU64 trajectory invariants; independent finer checks stratified, finite evidence; no execution compression in training labels'}


class Dataset:
    """Explicit index of independent continuous draws; no solver in GPU updates."""
    def __init__(self,path,certificate,*,progress=None):
        from .streams import PlannedReference,STREAM_SCHEMA
        self.path=Path(path);self.index=json.loads(self.path.read_text());self.config=DatasetConfig(**self.index['config'])
        require(self.index['schema']==SCHEMA and self.index.get('stream_schema')==STREAM_SCHEMA and self.index['dataset_identity']==self.config.identity,'dataset index/config mismatch')
        self.certificate=certificate;self.norms=normalizers(certificate);self.progress=progress or (lambda *a,**k:None)
        require(self.index['normalizer_identity']==self.norms.identity and self.index['certificate_id']==certificate['certification_id'],'dataset numerical capability mismatch')
        authorize(certificate,benchmark='G1_smooth_v1',pdes=self.config.pdes,families=self.config.families,normalizer_identity=self.norms.identity)
        require(self.index['original_split_identity']==split_manifest()['identity'],'benchmark splits changed')
        require(len(set(self.index['records'].values()))==len(self.index['records']),'physical case records alias across splits')
        require(self.index['inventory_identity']==digest(self.index['cases']),'physical inventory changed')
        self.root=self.path.parent.parent.parent;self.cache=PlannedReference(self);self.rows={r['case_id']:r for r in self.index['cases']};self.verified=set()
    def cases(self,split,*,training_only=False):
        return tuple(case_from_key(k,self.config.conditioning_h) for k in self.config.keys(split) if not training_only or k[1]!=self.config.holdout_family)
    def require_case(self,c):
        require(c.case_id in self.rows and self.rows[c.case_id]['key'][2]==c.split,'physical case not in dataset partition')
        if c.case_id not in self.verified:
            rel=self.index['records'].get(c.case_id);require(rel is not None,'unprepared or quarantined physical case: '+c.case_id)
            r,p=read_payload(self.root/rel,kind='diagnostic_bundle',required=('qc','case','labels','sampling_space'))
            require(r['case_id']==c.case_id and r['dataset_identity']==self.config.identity and r['normalizer_identity']==self.norms.identity and r['certificate_id']==self.certificate['certification_id'] and p['qc']['passed'],'case QC/identity failed')
            require(p['case']==c.manifest and r['physics']==self.cache.physics and r['execution_source']==self.cache.execution_source,'case/reference source changed')
            require((self.root/p['sampling_space']).is_file(),'sampling-space asset missing')
            self.cache.loaded[c.case_id]=p
            while len(self.cache.loaded)>self.cache.max_cases:self.cache.loaded.popitem(last=False)
            self.verified.add(c.case_id)
    def require_ready(self,split='train'):
        cases=self.cases(split,training_only=split=='train')
        for i,c in enumerate(cases):
            if i%32==0:self.progress('Dataset cache loading',split=split,checked=i,total=len(cases))
            self.require_case(c)
        for task in self.config.tasks:
            qualification=self.index['qualification'].get('/'.join(task),{})
            require(qualification.get('passed') is True and len(set(qualification.get('cases',[])))>=self.config.fine_spot_checks_per_task,'stratified finer-reference check missing/failed: '+str(task))
    def summary(self):
        return {'dataset_identity':self.config.identity,'target':self.config.totals,'prepared':len(self.index['records']),'quarantined':len(self.index['quarantine']),
                'stratified_checks':self.index['qualification'],'stream_schema':self.index['stream_schema'],'planned_updates':self.config.stream_updates,
                'reference_storage':'only exact draw labels + recipes, transient full trajectories discarded','scope':self.config.preview()['scope'],'manifest':str(self.path)}


def prepare_dataset(config,root,certificate,*,progress=None,persist=None,max_cases=None):
    """Explicit offline generation; same physical case/continuous law, bounded RAM.

    One original FULL trajectory/case is evaluated at independently pre-drawn
    encounter coordinates for the entire declared schedule, then discarded.
    Interrupted preparation reuses each exact verified case record by manifest.
    """
    from .streams import encounters,prepare_labels,save_sampling_space,STREAM_SCHEMA
    from ..learning_diagnostics.reference_cache import FixedReference
    from .provenance import reference_execution_identity
    root=Path(root);progress=progress or (lambda stage,**kw:print(stage,kw,flush=True));norms=normalizers(certificate)
    authorize(certificate,benchmark='G1_smooth_v1',pdes=config.pdes,families=config.families,normalizer_identity=norms.identity)
    path=root/'datasets'/config.identity/'index.json';rows=inventory(config);events=encounters(config,rows,progress=progress)
    if path.exists():
        index=json.loads(path.read_text());require(index['dataset_identity']==config.identity and index['certificate_id']==certificate['certification_id'],'incompatible explicit dataset manifest')
    else:
        index={'schema':SCHEMA,'stream_schema':STREAM_SCHEMA,'config':asdict(config),'dataset_identity':config.identity,'certificate_id':certificate['certification_id'],
            'normalizer_identity':norms.identity,'original_split_identity':split_manifest()['identity'],'cases':rows,'inventory_identity':digest(rows),
            'records':{},'assets':{},'quarantine':{},'qualification':{},'persistence':{},'reference_scope':'finite sampled certificate + per-case invariants + separate finer checks; exact pre-drawn continuous streams'}
        atomic_json(path,index)
    physics=__import__('pdeep.smooth.definition',fromlist=['physics_identity']).physics_identity();execution_source=reference_execution_identity();start=time.perf_counter();prepared=0
    # Geometry-major preparation reuses one heavy P4 assembly across all three PDEs.
    for p,g in ((p,g) for g in config.families for p in config.pdes):
        task='/'.join((p,g));required=sum(config.counts)
        for split,n in zip(('train','validation','test'),config.counts):
            for ordinal in range(n):
                c=Case(g,p,split,case_index(g,ordinal),config.conditioning_h)
                done=sum(1 for row in rows if row['key'][:2]==(p,g) and row['case_id'] in index['records'])
                progress('Dataset inspection',task=task,prepared=done,required=required,reference_failures=sum(r['task']==task for r in index['quarantine'].values()),elapsed_seconds=time.perf_counter()-start)
                if c.case_id in index['records']:read_payload(root/index['records'][c.case_id]);continue
                if c.case_id in index['quarantine']:continue
                if max_cases is not None and prepared>=max_cases:return Dataset(path,certificate,progress=progress)
                finer=split=='train' and ordinal<config.fine_spot_checks_per_task
                try:
                    progress('Reference preparation',task=task,case=c.case_id,method='original FULL CPU64 trajectory')
                    stem=digest((c.manifest,config.identity,execution_source,certificate['certification_id']))
                    record=root/'cases'/(stem+'.json')
                    if record.is_file():
                        # Recover a complete artifact even if disconnect preceded the
                        # mutable index commit. Never re-solve/overwrite completed work.
                        meta,payload=read_payload(record,kind='diagnostic_bundle',required=('case','qc','labels','sampling_space'))
                        require(meta['dataset_identity']==config.identity and meta['execution_source']==execution_source and payload['case']==c.manifest and payload['qc']['passed'],'incompatible committed case artifact')
                        qc=payload['qc'];space=root/payload['sampling_space'];read_manifest(space)
                        progress('Reference preparation',task=task,case=c.case_id,cache='verified committed record reuse; zero solves')
                    else:
                        full=FixedReference(c)
                        qc=quality_check_full(c,full,norms,config,finer=finer);require(qc['passed'],'per-case numerical QC failed: '+str(qc['checks']))
                        labels=prepare_labels(c,full,config,events[c.case_id],progress=progress);space=save_sampling_space(full,root,execution_source)
                        meta={'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'G1-unified-cases','scientific_schema':SCHEMA,'stream_schema':STREAM_SCHEMA,
                            'case_id':c.case_id,'dataset_identity':config.identity,'normalizer_identity':norms.identity,'certificate_id':certificate['certification_id'],'physics':physics,'execution_source':execution_source,
                            'reference_preparation_seconds':full.preparation_seconds,'transient_full_bytes':full.bytes,'stored_label_bytes':sum(v.numel()*v.element_size() for v in labels.values())}
                        payload={'case':c.manifest,'qc':qc,'labels':labels,'stream_events':events[c.case_id],'sampling_space':str(space.relative_to(root))}
                        record=write_artifact(root/'cases',stem,meta,payload,required=tuple(payload),idempotent=True)
                        del full,labels
                    index['records'][c.case_id]=str(record.relative_to(root));index['assets'][space.stem]=str(space.relative_to(root))
                    if finer:
                        old=index['qualification'].get(task,{'passed':True,'cases':[]})
                        index['qualification'][task]={'passed':old['passed'] and qc['passed'],'cases':list(dict.fromkeys([*old['cases'],c.case_id])),'policy':'independent P3/.01/.001 + fixed-space dt halving; separate finite dataset qualification'}
                    if persist:
                        if not index['persistence'].get(str(space),{}).get('Drive_verified'):index['persistence'][str(space)]=persist(space)
                        index['persistence'][str(record)]=persist(record)
                    del payload
                except Exception as error:
                    index['quarantine'][c.case_id]={'task':task,'key':(p,g,split,c.index),'error':f'{type(error).__name__}: {error}'}
                    progress('Reference quarantine',task=task,case=c.case_id,error=str(error))
                prepared+=1;atomic_json(path,index)
                if persist:persist(path,index_only=True)
    return Dataset(path,certificate,progress=progress)
