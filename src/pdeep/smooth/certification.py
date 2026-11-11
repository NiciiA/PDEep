"""Immutable, content-addressed numerical evidence and fail-closed capability gate."""
from pathlib import Path
import json
from ..core import require,digest
from .definition import *
from .data import Normalizers

ROOT=Path(__file__).resolve().parents[3]/'certifications'/BENCHMARK_ID
THRESHOLD=.002

def additional_sources():
    from ..artifact_io import sha256
    root=Path(__file__).resolve().parents[3]
    names=('src/pdeep/core.py','src/pdeep/wave_fidelity/study.py','src/pdeep/geometric/fem.py','docs/scientific_contract_v1.md','docs/scientific_contract_geometry_v1.md','docs/scientific_contract_geometry_g3_reference_v2.md')
    return {n:sha256(root/n) for n in names}

def save_record(body,directory=ROOT):
    body=dict(body);body.pop('certification_id',None);identifier=digest(body);record={**body,'certification_id':identifier}
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);path=directory/(identifier+'.json')
    data=json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if path.exists():require(path.read_text()==data,'immutable certification collision')
    else:
        with path.open('x') as stream:stream.write(data)
    return path

def load_record(path=None,*,current=True):
    path=ROOT/'default.json' if path is None else Path(path)
    require(path.is_file(),f'Selected benchmark is not numerically certified. Benchmark: {BENCHMARK_ID}. Failed or missing capability: certificate. Required verification: run scripts/certify_smooth.py; no training bypass exists.')
    record=json.loads(path.read_text())
    if 'record' in record:record=json.loads((path.parent/record['record']).read_text())
    body={k:v for k,v in record.items() if k!='certification_id'}
    require(digest(body)==record.get('certification_id'),'certification content hash mismatch')
    require(record['benchmark_id']==BENCHMARK_ID,'wrong certification benchmark')
    require(record.get('certification_schema')=='G1_smooth-numerical-certification-v1' and 'mesh_records' in record and record.get('independent_controls',{}).get('passed') and record.get('boundary_controls',{}).get('passed') and record.get('analytical_square_controls',{}).get('passed'),'Incomplete certification: mesh identities and independent controls are required')
    if current:require(record.get('additional_scientific_sources')==additional_sources(),'Scientific dependency source changed; recertification required')
    if current:require(record['source_revision']==physics_identity(),'Selected benchmark is not numerically certified: scientific source/dependency identity changed. Required verification: recertify changed generator/solver/environment.')
    return record

def fixed_time_spatial_bounds(record,pde,family):
    """Independent fixed-dt spatial bound from already measured differences.

    A=P4/.02/dt.002, C=P4/.02/dt.001, B=P3/.01/dt.001.
    ||C-B|| <= ||A-B|| + ||A-C||, separately per case/quantity/time/sensors.
    This is a conservative bound, NEVER labeled an actual measured difference.
    """
    rows=record['sampled_error_tables'][family];result=[]
    for q in ('u','v') if pde=='wave' else ('u',):
        for i in range(4):
            ab=next(r for r in rows if (r['pde'],r['quantity'],r['index'],r['comparison'])==(pde,q,i,'production_vs_independent'))
            ac=None if pde=='poisson' else next(r for r in rows if (r['pde'],r['quantity'],r['index'],r['comparison'])==(pde,q,i,'dt002_to001'))
            field=ab['field_normalized_RMSE']+(0 if ac is None else ac['field_normalized_RMSE'])
            sensor=ab['sensor_normalized_RMSE']+(0 if ac is None else ac['sensor_normalized_RMSE'])
            time={t:v+(0 if ac is None else ac['time_normalized_RMSE'][t]) for t,v in ab['time_normalized_RMSE'].items()}
            result.append({'pde':pde,'family':family,'quantity':q,'index':i,'field_RMSE_upper_bound':field,'sensor_RMSE_upper_bound':sensor,'time_RMSE_upper_bounds':time,'maximum_upper_bound':max(field,sensor,*time.values())})
    return result


def authorize(record,*,benchmark=BENCHMARK_ID,pdes=PDES,families=SEEN,policy=None,normalizer_identity=None,boundary=BOUNDARY_VERSION):
    require(benchmark==BENCHMARK_ID,f'Selected benchmark is not numerically certified. Benchmark: {benchmark}. Original rough G1 remains blocked; required verification: independent reference convergence.')
    # Recheck integrity/current source even when handed a decoded object.
    require(digest({k:v for k,v in record.items() if k!='certification_id'})==record.get('certification_id'),'certification content hash mismatch')
    require(record.get('certification_schema')=='G1_smooth-numerical-certification-v1' and record.get('independent_controls',{}).get('passed') and record.get('boundary_controls',{}).get('passed') and record.get('analytical_square_controls',{}).get('passed') and 'mesh_records' in record,'Incomplete numerical certification; independent controls and exact mesh identities required')
    require(record['source_revision']==physics_identity(),'Selected benchmark is not numerically certified: source/dependency identity changed; recertification required')
    require(record.get('additional_scientific_sources')==additional_sources(),'Scientific dependency source changed; recertification required')
    require(record.get('known_coefficients')==COEFFICIENTS,'Uncertified physical coefficients; source/configuration changed')
    require(record['reference_policy']==(POLICY if policy is None else policy),'Uncertified mesh/solver/prior/observation policy; rerun numerical certification')
    require(boundary in record['certified_boundary_policies'],'Uncertified boundary policy; mixed/Neumann research is not certified by this D0 record')
    require(normalizer_identity is None or normalizer_identity==record['normalizers']['identity'],'Uncertified normalization identity')
    capabilities={(c['pde'],c['family']) for c in record['capabilities'] if c['passed'] and c['physical_cases']>=4 and c['maximum_sampled_normalized_RMSE']<=THRESHOLD and max(r['maximum_upper_bound'] for r in fixed_time_spatial_bounds(record,c['pde'],c['family']))<=THRESHOLD}
    for family in families:
        if family in record.get('mesh_records',{}):
            expected=record['mesh_records'][family]['production']
            require(geometry(family,POLICY['reference_h']).mesh_id==expected['mesh_id'],'Selected reference mesh identity differs from certified mesh; recertification required')
    missing=set((p,g) for p in pdes for g in families)-capabilities
    require(not missing,f'Selected benchmark is not numerically certified. Benchmark: {benchmark}. Failed or missing capability: {sorted(missing)}. Required verification: four fixed prior cases, independent space/time/sensor/boundary checks below0.2% training RMS.')
    return {'certification_id':record['certification_id'],'capabilities':sorted(capabilities),'status':'certified for requested finite numerical regime'}

def normalizers(record):return Normalizers.fromdict(record['normalizers'])
