"""Non-optimizing smooth benchmark regression and prepared-experiment checks."""
from pathlib import Path
from dataclasses import replace,asdict
import json,copy
import numpy as np
import pytest
import torch
from pdeep.core import ValidationError,digest
from pdeep.smooth.definition import *
from pdeep.smooth.data import Case,case,design,tokens,Normalizers
from pdeep.smooth.certification import load_record,authorize,save_record,THRESHOLD
from pdeep.smooth.training import Config,Engine,resume,implementation_identity
from pdeep.smooth.bank import Bank
from pdeep.wave_fidelity.fields import pair

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def certificate():return load_record()


def test_unchanged_wave_and_authorized_extension():
    for family in GEOMETRIES:
        m=geometry(family,.08);old=pair(m,('validation',2));fields=physical_fields(family,'wave','validation',2)
        assert fields['u0'].identity==old[0].identity and fields['v0'].identity==old[1].identity
        for p in ('poisson','heat'):
            f=next(iter(physical_fields(family,p,'train',0).values()))
            assert f.specification['version']==EXTENSION_VERSION
            assert np.array_equal(f.centers,old[0].centers) and f.radius==old[0].radius
            assert f.identity!=old[0].identity
            assert np.max(abs(f(m.vertices[np.unique(m.boundary_edges)])))==0


def test_full_split_overlap_and_remesh():
    sets={}
    for split,count in SPLIT_COUNTS.items():
        families=HELD_OUT if split=='geometry_held_out' else SEEN
        ids={physical_id(families[i%len(families)],p,split,i) for p in PDES for i in range(count)}
        assert len(ids)==len(PDES)*count;sets[split]=ids
    for a in sets:
        for b in sets:
            if a!=b:assert sets[a].isdisjoint(sets[b])
    a=case('train','wave',0,conditioning_h=.12);b=case('train','wave',0,conditioning_h=.08)
    assert a.case_id==b.case_id and a.problem.mesh.mesh_id!=b.problem.mesh.mesh_id
    with pytest.raises(ValidationError,match='leakage'):case('train','wave',0,family='hole')
    for name in ('u0','v0'):assert a.fields[name].identity==b.fields[name].identity


def test_independent_observation_streams_and_stationary_time(certificate):
    from pdeep.smooth.certification import normalizers
    n=normalizers(certificate)
    for pde in PDES:
        c=case('train',pde,0);coords=design(c,64,observation=True,set_id=3);clean=np.linspace(-.1,.1,64)
        a,meta=tokens(c,64,n,observation=True,set_id=3,coordinates=coords,clean=clean)
        b,mb=tokens(c,64,n,observation=True,set_id=3,coordinates=coords,clean=clean)
        assert torch.equal(a.value,b.value) and a.sigma.min()>0 and a.valid.all()
        expected=meta['sigma']*random_stream(c.case_id,'measurement_noise',3).standard_normal(64)
        np.testing.assert_allclose(a.value[0,:,0].numpy(),clean+expected,atol=1e-7,rtol=1e-6)
        assert torch.equal(a.time_present,torch.full_like(a.valid,pde!='poisson'))
        assert not np.array_equal(random_stream(c.case_id,'noise_levels',3).random(10),random_stream(c.case_id,'measurement_noise',3).random(10))
        if pde=='wave':assert int((a.quantity_id==0).sum())==int((a.quantity_id==1).sum())==32
        if pde=='poisson':
            with pytest.raises(ValidationError,match='time'):c.evaluate(coords[0],np.zeros(64))


def test_certificate_all_capabilities_and_immutable_integrity(certificate,tmp_path):
    assert len(certificate['validation_case_ids'])==96
    assert len(certificate['capabilities'])==24
    assert certificate['source_revision']==physics_identity()
    assert all(c['physical_cases']==4 for c in certificate['capabilities'])
    for p in PDES:
        for family in GEOMETRIES:
            cap=next(c for c in certificate['capabilities'] if (c['pde'],c['family'])==(p,family))
            if cap['passed']:authorize(certificate,pdes=(p,),families=(family,))
            else:
                with pytest.raises(ValidationError,match='capability'):authorize(certificate,pdes=(p,),families=(family,))
    path=save_record(certificate,tmp_path);assert load_record(path)['certification_id']==certificate['certification_id']
    bad=copy.deepcopy(certificate);bad['reference_policy']['reference_h']=.04
    with pytest.raises(ValidationError,match='hash'):authorize(bad)
    with pytest.raises(ValidationError,match='rough'):authorize(certificate,benchmark='G1')
    with pytest.raises(ValidationError,match='mesh'):authorize(certificate,policy={**POLICY,'reference_h':.04})
    with pytest.raises(ValidationError,match='boundary'):authorize(certificate,boundary='mixed')
    with pytest.raises(ValidationError,match='normalization'):authorize(certificate,normalizer_identity='bad')
    with pytest.raises(ValidationError,match='capability'):authorize(certificate,families=('unknown-instance',))
    bad=copy.deepcopy(certificate);bad['source_revision']['scientific_sha']='changed';bad['certification_id']=digest({k:v for k,v in bad.items() if k!='certification_id'})
    with pytest.raises(ValidationError,match='source'):authorize(bad)


def test_rough_gate_intact():
    from pdeep.geometric.training import Engine as RoughEngine
    from types import SimpleNamespace
    with pytest.raises(ValidationError,match='BLOCKED'):RoughEngine.run(SimpleNamespace(config=SimpleNamespace(preset='bounded')))


def test_manual_configs_preserve_real_losses_and_existing_loop(certificate):
    from pdeep.training import TrainingEngine
    from pdeep.geometric.physics import Adapter
    assert Engine.__mro__[1] is TrainingEngine
    for a in ('A_GEO','C_GEO'):
        tiny=Config(architecture=a,preset='tiny_overfit',updates=100,training_families=('square','rectangle'),cases_per_geometry=1,device='cpu')
        assert len(tiny.case_pool)==6 and tiny.scheduler=='constant'
        b=Config(architecture=a,updates=1000,device='cpu');assert b.updates==1000 and b.case_pool is None and b.case_pool_size==36000
        assert b.checkpoint_every==b.validation_every==250
        assert (b.spatial_order,b.time_order,b.weak_tests)==(3,3,16)
        assert tiny.architecture==a
    with pytest.raises(ValidationError,match='budget'):Config(updates=10000,device='cpu')
    with pytest.raises(ValidationError):Config(benchmark='G1',device='cpu')


def test_query_bank_design_is_reproducible_and_input_safe(certificate):
    from pdeep.smooth.bank import grid
    from pdeep.geometric.data import Problem
    for p in PDES:
        c=case('train',p,0);a=design(c,128);b=design(c,128)
        for aa,bb in zip(a,b):assert np.array_equal(aa,bb)
        assert c.problem.mesh.features(a[0])['inside'].all()
        assert isinstance(c.problem,Problem) and not hasattr(c.problem,'fields')
        v=grid(c,8);assert c.problem.mesh.features(v[0])['inside'].all()
        if p!='poisson':assert set(np.round(v[1],5))=={0,.02,.05,.1,.2,.35,.5,.75,1}


def test_notebook_single_launch_and_no_legacy_path():
    n=json.loads((ROOT/'notebooks/02_train.ipynb').read_text());code='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
    assert 'RUN_MODE = "preview"' in code and 'MODELS = ("A_GEO", "C_GEO")' in code
    assert code.count('offer_launch(')==1
    assert 'legacy_training' not in code and 'GEO_ENGINE' not in code 
    assert 'RESULTS.update(launch(PLAN' in code
    import ast
    # Launch is inside callback only, never an executed top-level expression.
    tree=ast.parse(code)
    for statement in tree.body:
        if isinstance(statement,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
        for call in ast.walk(statement):
            if isinstance(call,ast.Call):
                name=call.func.id if isinstance(call.func,ast.Name) else call.func.attr if isinstance(call.func,ast.Attribute) else ''
                assert name not in {'Engine','run','update','launch','step'}


def test_colab_pinning_local_is_noop(tmp_path,monkeypatch):
    from pdeep.smooth.bootstrap import prepare_reference_environment
    def forbidden(*a,**k):raise AssertionError('local must never pip install or import Drive')
    r=prepare_reference_environment(tmp_path,is_colab=False,runner=forbidden)
    assert not r['installed']


def test_engine_rejects_before_optimizer_constructor(certificate,tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:calls.append(1))
    broken=copy.deepcopy(certificate);broken['source_revision']['scientific_sha']='changed';broken['certification_id']=digest({k:v for k,v in broken.items() if k!='certification_id'})
    with pytest.raises(ValidationError,match='source'):Engine(Config(device='cpu'),tmp_path,certificate=broken)
    assert not calls and not list(tmp_path.iterdir())


def test_fixed_physical_observations_survive_conditioning_remesh():
    from pdeep.smooth.bank import physical_design
    a=Case('hole','wave','geometry_held_out',3,.12);b=replace(a,conditioning_h=.08)
    for observation in (False,True):
        x=physical_design(a,64,observation=observation,set_id=9);y=physical_design(b,64,observation=observation,set_id=9)
        assert all(np.array_equal(v,w) for v,w in zip(x,y))


def test_notebook_tiny_budget_uses_overfit_setting():
    import ast
    n=json.loads((ROOT/'notebooks/02_train.ipynb').read_text());scope={}
    exec(''.join(n['cells'][4]['source']),scope)
    scope.update(IS_COLAB=False,PERSISTENCE=None,RUN_NAME='budget-fixture',ADVANCED={},
        SETUP={'local_output_dir':'/tmp/fixture'},SETUP_CONFIGURATION=('url','ref','project',False,'drive','local'),
        REPO_URL='url',REPO_REF='ref',PROJECT_DIR='project',USE_GOOGLE_DRIVE=False,DRIVE_ROOT='drive',LOCAL_OUTPUT_DIR='local')
    code='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
    definition=next(node for node in ast.parse(code).body if isinstance(node,ast.FunctionDef) and node.name=='training_settings')
    exec(compile(ast.Module(body=[definition],type_ignores=[]),'<notebook-budget>','exec'),scope)
    assert scope['training_settings']()['updates']==100
    scope['EXPERIMENT']='bounded';assert scope['training_settings']()['updates']==1000
    scope['EXPERIMENT']='tiny_overfit';assert scope['training_settings']()['updates']==100
    scope['OVERFIT_UPDATES']=25;assert scope['training_settings']()['updates']==25


def test_colab_preloaded_stack_no_longer_requires_pin_or_restart():
    from pdeep.smooth.bootstrap import prepare_reference_environment
    result=prepare_reference_environment(ROOT,is_colab=False)
    assert not result['installed']
