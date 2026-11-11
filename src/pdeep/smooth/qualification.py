"""Explicit bounded, model-independent qualification of a numerical environment.

Reuses the scientific implementation. Four fixed cases/PDE/geometry are compared
against frozen CPU64 values from the original valid certificate. Its independent
space/time error bounds transfer by the triangle inequality; current-runtime
roundoff/solver differences must be <=1e-6 TRAIN RMS (pointwise <=1e-5).
No model construction, optimizer, training, interpolation or validation calibration.
"""
from pathlib import Path
from functools import lru_cache
import hashlib,json,platform,time
import numpy as np
from ..core import digest,require
from ..geometric.portable import index,ROOT as INPUTS,sha
from .definition import *
from .resolution import resolve_reference,report_selection

CACHE=Path(__file__).resolve().parents[3]/'artifacts/reference_certificates'
DELTA_RMS=1e-6
DELTA_MAX=1e-5

def runtime_identity():
    from .certification import additional_sources
    root=Path(__file__).resolve().parents[3]
    extra=('src/pdeep/geometric/portable.py','src/pdeep/smooth/resolution.py','src/pdeep/smooth/qualification.py')
    return {'physics':physics_identity(),'additional':additional_sources(),'resolution_sources':{p:sha(root/p) for p in extra},'portable_inputs':index()['identity'],'platform':platform.system()+'/'+platform.machine(),'python':platform.python_version()}

def _validate_implementation(parent):
    catalog=index();require(catalog is not None,'portable certified numerical inputs are missing; push certifications/G1_smooth_v1/portable')
    require(parent['certification_id']==catalog['parent_certification_id'],'portable inputs belong to another scientific certificate')
    approved=json.loads((INPUTS/'implementation.json').read_text())
    require(approved['identity']==digest({k:v for k,v in approved.items() if k!='identity'}),'implementation compatibility proof checksum mismatch')
    current=physics_identity()['files'];require(current==approved['physics_files'],'scientific generator/solver implementation changed; full scientific recertification is required, not environment qualification')
    require(parent['source_revision']['files']==approved['parent_physics_files'],'wrong parent implementation evidence')
    return catalog

def cache_path(parent):return CACHE/(digest((parent['certification_id'],runtime_identity()))+'.json')

def cached_record(parent):
    from .certification import load_record
    path=cache_path(parent)
    if path.exists():return load_record(path),path
    # Shipped native-qualified records are immutable and independently verified.
    shipped=Path(__file__).resolve().parents[3]/'certifications/G1_smooth_v1/qualified'
    if shipped.exists():
        for candidate in sorted(shipped.glob('*.json')):
            try:
                r=load_record(candidate)
                if r.get('runtime_identity')==runtime_identity():return r,candidate
            except ValueError:continue
    return None,None

def ensure_compatible_certificate(*,path=None,progress=print):
    from .certification import load_record,save_record,normalizers,authorize,fixed_time_spatial_bounds,additional_sources,THRESHOLD
    if path is not None:
        record=load_record(path);authorize(record);progress('Reusing selected valid certificate '+record['certification_id']);return record,Path(path)
    parent=load_record(current=False);_validate_implementation(parent)
    cached,location=cached_record(parent)
    if cached is not None:
        authorize(cached);progress('Reusing numerically qualified environment certificate '+cached['certification_id']);report_selection(cached,SEEN,progress=progress);return cached,location
    progress('No compatible environment certificate. Starting bounded CPU64 qualification:96 physical cases; zero optimizer updates.')
    progress('Python '+__import__('sys').executable+'; numerical versions '+str(physics_identity()['dependencies']))
    report_selection(parent,SEEN,progress=progress);norms=normalizers(parent);catalog=index();rows=[];resolved={};start=time.perf_counter()
    from .reference import Reference
    for family in GEOMETRIES:
        info=catalog['families'][family];fixture=INPUTS/info['reference_file'];require(sha(fixture)==info['reference_sha256'],'frozen reference fixture checksum mismatch')
        with np.load(fixture,allow_pickle=False) as data:
            x=data['position'];t=data['time'];expected={k:data[k].copy() for k in data.files if k not in {'position','time'}}
        mesh,spec,origin=resolve_reference(family);require(mesh.mesh_id==parent['mesh_records'][family]['production']['mesh_id'],'selected mesh differs from parent; run explicit spatial/temporal recertification for this changed specification')
        resolved[family]=spec.asdict();r=Reference(family,spec.h,spec.degree);split='validation' if family in SEEN else 'geometry_held_out'
        for p in PDES:
            progress(f'Qualification {family}/{p}:4 fixed cases, P{spec.degree}, h={spec.h}, dt={spec.dt}; elapsed {time.perf_counter()-start:.1f}s')
            fields=[physical_fields(family,p,split,i) for i in range(4)];identities=[{k:v.identity for k,v in f.items()} for f in fields]
            require(identities==info['field_identities'][p],'C3 physical prior draws differ from the frozen generator; refuse numerical qualification')
            if p=='poisson':u,diagnostic=r.poisson([f['f'] for f in fields],x,spec.source_quadrature);v=None
            else:u,v,diagnostic=r.dynamic(p,[f['u0'] for f in fields],[f['v0'] for f in fields] if p=='wave' else None,x,t,spec.dt)
            for q,values in [('u',u)]+([('v',v)] if v is not None else []):
                delta=(values-expected[p+'_'+q])/norms.scale(p,q);rmse=np.sqrt(np.mean(delta*delta,axis=1));maximum=float(abs(delta).max())
                require(np.isfinite(delta).all() and float(rmse.max())<=DELTA_RMS and maximum<=DELTA_MAX,f'Environment numerical comparison failed: {family}/{p}/{q}; max case RMS={rmse.max()}, max point={maximum}. Training stays blocked.')
                rows.append({'family':family,'pde':p,'quantity':q,'case_RMSE_differences':rmse.tolist(),'max_case_RMSE_difference':float(rmse.max()),'max_point_difference':maximum,'diagnostic':diagnostic})
        del r
    # Bound NEW production error against the SAME independent parent comparator.
    capabilities=[]
    for c in parent['capabilities']:
        delta=max(r['max_case_RMSE_difference'] for r in rows if (r['family'],r['pde'])==(c['family'],c['pde']))
        conservative=c['maximum_sampled_normalized_RMSE']+max(r['max_point_difference'] for r in rows if (r['family'],r['pde'])==(c['family'],c['pde']))
        spatial=max(r['maximum_upper_bound'] for r in fixed_time_spatial_bounds(parent,c['pde'],c['family']))+delta
        require(conservative<=THRESHOLD and spatial<=THRESHOLD,'inherited reference budget plus current environment discrepancy exceeds tolerance')
        capabilities.append({**c,'maximum_sampled_normalized_RMSE':conservative,'current_environment_RMSE_delta':delta,'passed':c['passed'] and conservative<=THRESHOLD and spatial<=THRESHOLD})
    body={k:v for k,v in parent.items() if k!='certification_id'}
    body.update(source_revision=physics_identity(),additional_scientific_sources=additional_sources(),parent_certification_id=parent['certification_id'],runtime_identity=runtime_identity(),resolved_meshes=resolved,portable_inputs_id=catalog['identity'],capabilities=capabilities,
        qualification={'method':'production-to-frozen-parent comparison; four cases/PDE/geometry,128 field points x9 times plus128 sensors. Parent independently refined bounds transfer by triangle inequality. Historical parent controls are inherited evidence, not rerun measurements.','rows':rows,'max_RMS_delta':max(r['max_case_RMSE_difference'] for r in rows),'max_point_delta':max(r['max_point_difference'] for r in rows),'seconds':time.perf_counter()-start,'optimizer_updates':0},
        verification_timestamp=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    body['numerical_error_summary']={'maximum':max(c['maximum_sampled_normalized_RMSE'] for c in capabilities),'threshold':THRESHOLD,'meaning':'conservative parent reference bound + measured current-runtime discrepancy'}
    target=save_record(body,CACHE);pointer=cache_path(parent)
    # Alias is a tiny pointer, never an overwritten immutable certificate.
    pointer.write_text(json.dumps({'record':target.name})+'\n');record=load_record(target);authorize(record)
    progress(f'Qualification passed in {time.perf_counter()-start:.1f}s; certificate {record["certification_id"]}; cached at {target}')
    return record,target
