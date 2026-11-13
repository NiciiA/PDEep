"""Safe legacy loading and exact Study-A resume; never takes an optimizer step."""
from dataclasses import asdict
from pathlib import Path
import copy,json,pickle,zipfile
import numpy as np
import pytest
import torch
from pdeep.core import digest,ValidationError
from pdeep.artifact_io import read_manifest,read_payload,write_artifact,sha256
from pdeep.artifact_tensors import native_scalars,tree_identity,state_identity

@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    def forbidden(*a,**kw):pytest.fail('No training/recertification allowed')
    for cls in (torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD):monkeypatch.setattr(cls,'step',forbidden)
    monkeypatch.setattr('pdeep.smooth.qualification.ensure_compatible_certificate',forbidden)


def project_metadata():
    from pdeep.smooth.training import implementation_identity
    from pdeep.learning_diagnostics.engine import diagnostic_source,SCHEMA
    return {'artifact_type':'training_checkpoint','architecture_id':'C_GEO','run_id':'scalar-fixture','benchmark':'G1_smooth_v1','scientific_schema':SCHEMA,
        'implementation_identity':implementation_identity(),'diagnostic_source_id':diagnostic_source(),'config':{}}


def legacy_artifact(root,payload,metadata=None,old_numpy_path=False):
    """Emit old project bytes + verified manifest; never unsafe-load the fixture."""
    root.mkdir(parents=True,exist_ok=True);p=root/'legacy.pt';torch.save(payload,p,pickle_protocol=2)
    if old_numpy_path:
        with zipfile.ZipFile(p) as z:files={n:z.read(n) for n in z.namelist()}
        with zipfile.ZipFile(p,'w') as z:
            for n,data in files.items():z.writestr(n,data.replace(b'numpy._core.multiarray\nscalar\n',b'numpy.core.multiarray\nscalar\n') if n.endswith('data.pkl') else data)
    metadata=copy.deepcopy(metadata or project_metadata())
    excluded={'schema_version','content_identity','payload_tree_identity','created_UTC','files','verification','artifact_id'}
    metadata={k:v for k,v in metadata.items() if k not in excluded}
    tree=digest(tree_identity(payload))
    body={**metadata,'schema_version':'pdeep-artifact-v2','payload_tree_identity':tree,'created_UTC':'2026-11-12T00:00:00+00:00',
        'content_identity':digest({'metadata':metadata,'payload_tree_identity':tree}),
        'files':{p.name:{'sha256':sha256(p),'size_bytes':p.stat().st_size}},'verification':{'status':'verified','local_payload_readback':True,'checksum_verified':True}}
    manifest=root/'legacy.json';manifest.write_text(json.dumps({**body,'artifact_id':digest(body)},sort_keys=True))
    return manifest

@pytest.mark.parametrize('old_numpy_path',[False,True])
def test_legacy_numeric_scalars_and_dynamic_dtypes_scoped(tmp_path,old_numpy_path):
    values=[np.float64(.49),np.float32(.1),np.float16(.2),np.int64(-7),np.uint64(2**63+1),np.bool_(True)]
    path=legacy_artifact(tmp_path,{'metadata':values},old_numpy_path=old_numpy_path)
    payload_file=path.parent/next(iter(read_manifest(path)['files']))
    with pytest.raises(pickle.UnpicklingError,match='numpy'):torch.load(payload_file,weights_only=True)
    before=list(torch.serialization.get_safe_globals());_,p=read_payload(path)
    assert [type(v) for v in p['metadata']]==[type(v) for v in values]
    assert all(a==b for a,b in zip(p['metadata'],values))
    assert set(torch.serialization.get_safe_globals())==set(before)
    # The scope is closed, including after a successful legacy load.
    with pytest.raises(pickle.UnpicklingError):torch.load(payload_file,weights_only=True)


def test_new_writes_are_primitive_and_hash_compatible(tmp_path):
    values={'floor':{'pointwise':{'scales':{'dynamic':np.float64(.49)}}},'count':np.int64(3),'flag':np.bool_(True),'label':np.str_('floor'),
            'tensor':torch.arange(4),'tuple':(np.float32(.25),{np.int64(2):np.uint64(7)})}
    p=write_artifact(tmp_path,'new',{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'regression','cases':np.int64(3)},values)
    record=read_manifest(p);payload=torch.load(p.parent/next(iter(record['files'])),weights_only=True)
    assert type(payload['floor']['pointwise']['scales']['dynamic']) is float
    assert type(payload['count']) is int and type(payload['flag']) is bool and type(payload['label']) is str
    assert type(next(iter(payload['tuple'][1]))) is int
    assert isinstance(payload['tuple'],tuple) and torch.equal(payload['tensor'],values['tensor'])
    assert not torch.serialization.get_unsafe_globals_in_checkpoint(p.parent/next(iter(record['files'])))
    assert digest(tree_identity({'x':np.float64(.49)}))==digest(tree_identity({'x':.49}))


def test_checksum_source_and_other_globals_stay_fail_closed(tmp_path,monkeypatch):
    good=legacy_artifact(tmp_path/'good',{'x':np.float64(.49)})
    calls=[];original=torch.load
    monkeypatch.setattr(torch,'load',lambda *a,**kw:calls.append(kw) or original(*a,**kw))
    record=read_manifest(good);p=good.parent/next(iter(record['files']));p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValidationError,match='size/checksum'):read_payload(good)
    assert not calls
    meta=project_metadata();meta['implementation_identity']='unapproved-source'
    bad=legacy_artifact(tmp_path/'source',{'x':np.float64(.49)},meta)
    with pytest.raises(ValidationError,match='implementation/source'):read_payload(bad)
    assert all(c.get('weights_only') is True for c in calls)
    # Generic/foreign manifests cannot opt in merely by having a valid checksum.
    other=legacy_artifact(tmp_path/'foreign',{'x':np.float64(.49)},{'artifact_type':'diagnostic_bundle','run_id':'foreign','architecture_id':'OTHER'})
    with pytest.raises(ValueError,match='restricted'):read_payload(other)


def test_arrays_and_object_dtype_not_allowlisted(tmp_path):
    from pdeep.artifact_numpy_compat import _checked_dtype,_checked_scalar
    before=set(torch.serialization.get_safe_globals())
    for value in ('O','V16','c16','f16'):
        with pytest.raises(ValidationError):_checked_dtype(value)
    with pytest.raises(ValidationError):_checked_scalar(np.dtype('float64'),b'bad length')
    # A raw ndarray requires _reconstruct/ndarray globals, intentionally excluded.
    p=legacy_artifact(tmp_path,{'array_tensor':torch.ones(1)})
    data={'array':np.arange(3)};target=p.parent/'legacy.pt';torch.save(data,target)
    record=json.loads(p.read_text());record['files']['legacy.pt']={'sha256':sha256(target),'size_bytes':target.stat().st_size};record.pop('artifact_id')
    p.write_text(json.dumps({**record,'artifact_id':digest(record)}))
    with pytest.raises(ValidationError,match='unsupported globals'):read_payload(p)
    assert set(torch.serialization.get_safe_globals())==before


FIXTURE=Path(__file__).resolve().parent/'fixtures/phase_g_legacy_numpy'

def existing_study():
    checkpoint=FIXTURE/'legacy.json'
    assert checkpoint.is_file(), 'portable existing-checkpoint regression fixture missing'
    meta=read_manifest(checkpoint)
    if meta['runtime']['dependencies']['torch']!=torch.__version__:
        pytest.skip('Exact-resume fixture requires its recorded Torch version; the loader must not bypass that guard')
    from pdeep.learning_diagnostics.certificates import resolve_certification
    current=resolve_certification(requested_capabilities={'certification_id':None})
    if not current.verified or current.record['certification_id']!=meta['execution_qualification']['certification_id']:
        pytest.skip('Exact-resume fixture requires the existing matching numerical runtime certificate; no recertification')
    return checkpoint


def test_existing_checkpoint_exact_resume_and_native_resave(tmp_path):
    checkpoint=existing_study();meta,original=read_payload(checkpoint,kind='training_checkpoint')
    payload=copy.deepcopy(original)
    payload['training_state']['validation_history'][0]['value']=np.float64(payload['training_state']['validation_history'][0]['value'])
    legacy=legacy_artifact(tmp_path/'old-bytes',payload,meta)
    before=sha256(checkpoint.parent/next(iter(meta['files'])))
    _,loaded=read_payload(legacy)
    from pdeep.learning_diagnostics.data import FixedData
    from pdeep.geometric.data import Tokens
    from pdeep.smooth.data import Case
    from pdeep.smooth.certification import normalizers
    from pdeep.learning_diagnostics.engine import DiagnosticConfig,resume_diagnostic
    from pdeep.learning_diagnostics.dashboard import Dashboard
    from pdeep.artifact_rng import capture_rng,restore_rng
    c=meta['config'];cfg=DiagnosticConfig(**{**c,**{k:tuple(c[k]) for k in ('pdes','probabilities','training_families')}})
    fixture_data=FIXTURE/json.loads((FIXTURE/'provenance.json').read_text())['frozen_data']
    _,d=read_payload(fixture_data,kind='diagnostic_bundle')
    case=Case(c['training_families'][0],c['pdes'][0],'train',meta['fixed_case']['index'],c['conditioning_h'])
    norms=normalizers(meta['execution_qualification'])
    data=FixedData(case,*[Tokens(**d[k]) for k in ('observation','training','diagnostic')],d['clean_observations'],norms,d['identities'],str(fixture_data))
    rng=capture_rng()
    try:
        e=resume_diagnostic(legacy,tmp_path/'resumed',data=data,certificate=meta['execution_qualification'],config=cfg,monitor=Dashboard(notebook=False))
        assert e.state.successful==original['training_state']['successful'] and e.state.sampler_cursor==original['training_state']['sampler_cursor']
        assert digest(tree_identity(e.optimizer.state_dict()))==digest(tree_identity(original['optimizer']))
        assert digest(tree_identity(e.scheduler.state_dict()))==digest(tree_identity(original['scheduler']))
        assert digest(tree_identity(capture_rng(e.rng_generators)))==digest(tree_identity(original['rng']))
        assert e.sampler.state_dict(e.state.sampler_cursor)==original['sampler']
        assert state_identity(e.model)==digest(tree_identity(original['model']))
        saved=e.save_checkpoint();record,clean=read_payload(saved)
        assert type(clean['training_state']['validation_history'][0]['value']) is float
        assert not torch.serialization.get_unsafe_globals_in_checkpoint(saved.parent/next(iter(record['files'])))
        for key in ('model','optimizer','scheduler','rng','sampler'):assert digest(tree_identity(clean[key]))==digest(tree_identity(original[key]))
    finally:restore_rng(rng)
    assert sha256(checkpoint.parent/next(iter(meta['files'])))==before


def test_study_A_resume_skips_completed_arms_and_writes_scalar_floor(tmp_path,monkeypatch):
    checkpoint=existing_study();original=read_manifest(checkpoint)
    from pdeep.learning_diagnostics.optimization_studies import prepare_study,run_study
    from pdeep.learning_diagnostics.certificates import resolve_certification
    from pdeep.learning_diagnostics.engine import DiagnosticConfig,diagnostic_source
    certificate=resolve_certification(requested_capabilities={'certification_id':None}).require()
    # Explicit one-arm regression study reuses the actual prior FIELD_ONLY
    # checkpoint; no other checkpoint/weight is fabricated or trained here.
    kwargs=dict(certificate=certificate,root=tmp_path/'restored',capacity='verification',device='cpu',query_count=32,query_chunk=16,
        updates=1,amp_mode='off',advanced={'grid_size':4,'validation_every':1,'checkpoint_every':1},variants=['FIELD_ONLY'])
    proposed=prepare_study('c_wave',**kwargs)
    c=original['config'];cfg=DiagnosticConfig(**{**c,**{k:tuple(c[k]) for k in ('pdes','probabilities','training_families')}})
    run=json.loads((FIXTURE/'original_run.json').read_text())
    run.update(checkpoint=str(checkpoint),checkpoint_id=original['artifact_id'],Drive_checkpoint=None,config_identity=cfg.identity,run_id=cfg.run_id)
    run['report']['stability']['elapsed_seconds']=np.float64(run['report']['stability']['elapsed_seconds'])
    state={'schema':'G1-smooth-diagnostic-suite-v1','suite_id':proposed.suite_id,'configuration_hash':proposed.identity,'certificate':certificate['certification_id'],
        'order':['c_physics_ablation'],'entries':{'c_physics_ablation':{'experiment':'c_physics_ablation','status':'completed','runs':{'FIELD_ONLY':run}}},
        'active_experiment':None,'revision':0,'status':'completed','preview':proposed.preview(),'source':diagnostic_source()}
    metadata={'artifact_type':'diagnostic_bundle','architecture_id':'C_GEO','run_id':proposed.suite_id,'suite_schema':state['schema'],'configuration_hash':proposed.identity}
    legacy=legacy_artifact(tmp_path/'suite-legacy',{'suite_state':state,'files':{}},metadata)
    plan=prepare_study('c_wave',**kwargs,resume_manifest=legacy,action='resume')
    monkeypatch.setattr('pdeep.learning_diagnostics.suite.prepare_diagnostic',lambda *a,**kw:pytest.fail('Completed variants must not be prepared/trained again'))
    # Only the post-study numerical computation is stubbed. Its real scalar
    # return shape reproduces the exact save-readback failure without PDE solves.
    monkeypatch.setattr('pdeep.learning_diagnostics.data.prepare_data',lambda *a,**kw:object())
    floor={'pointwise':{'scales':{'dynamic':np.float64(.49)}}}
    monkeypatch.setattr('pdeep.learning_diagnostics.consistency.reference_floor',lambda *a,**kw:(floor,None))
    result=run_study(plan,run_mode='resume',notebook=False)
    assert result['status']=='completed' and result['entries']['c_physics_ablation']['runs']['FIELD_ONLY']['last_successful_update']==1
    saved=Path(result['snapshot']).parent.parent/'diagnostics/reference-residual-floor.json'
    r=read_manifest(saved);p=torch.load(saved.parent/next(iter(r['files'])),weights_only=True)
    assert type(p['floor']['pointwise']['scales']['dynamic']) is float


def test_dtype_failure_restores_scope_and_arbitrary_globals_rejected(tmp_path):
    path=legacy_artifact(tmp_path,{'value':np.float64(.49)})
    def replace_bytes(payload):
        target=path.parent/'legacy.pt';torch.save(payload,target)
        record=json.loads(path.read_text());record.pop('artifact_id');record['files']['legacy.pt']={'sha256':sha256(target),'size_bytes':target.stat().st_size}
        path.write_text(json.dumps({**record,'artifact_id':digest(record)}))
    before=set(torch.serialization.get_safe_globals())
    replace_bytes({'value':np.float64(.49),'bad_dtype':np.dtype('O')})
    with pytest.raises(ValidationError,match='unsupported legacy NumPy dtype'):read_payload(path)
    assert set(torch.serialization.get_safe_globals())==before
    marker=tmp_path/'MUST_NOT_EXECUTE'
    class Untrusted:
        def __reduce__(self):return eval,(f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')",)
    replace_bytes({'value':np.float64(.49),'untrusted':Untrusted()})
    with pytest.raises(ValidationError,match='unsupported globals'):read_payload(path)
    assert not marker.exists() and set(torch.serialization.get_safe_globals())==before


def test_numpy_dtype_class_compatibility_with_existing_scoped_scalar_allowlist(tmp_path):
    core=getattr(np,'_core',None)
    if core is None:core=np.core
    path=legacy_artifact(tmp_path,{'value':np.float64(.49)})
    # Reproduce the second error when a notebook already allowed scalar/dtype
    # functions, but omitted dynamically built Float64DType on NumPy>=1.25.
    with torch.serialization.safe_globals([np.dtype,core.multiarray.scalar]):
        before=set(torch.serialization.get_safe_globals())
        _,payload=read_payload(path)
        assert payload['value']==.49 and set(torch.serialization.get_safe_globals())==before
