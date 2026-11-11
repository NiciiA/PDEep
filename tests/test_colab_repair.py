"""Setup/mesh regression. Real numerical profiles are verified by separate runs."""
import ast,copy,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from pdeep.core import digest,ValidationError
from pdeep.smooth import bootstrap
from pdeep.smooth.definition import GEOMETRIES,POLICY,physics_identity
from pdeep.smooth.resolution import resolve_reference,report_selection
from pdeep.smooth.certification import load_record,authorize
from pdeep.geometric.domain import make_shape,mesh_shape

ROOT=Path(__file__).resolve().parents[1]

def installed_profile():
    return {name:{'installed':version,'loaded':version} for name,version in {'numpy':'2.4.6','scipy':'1.16.3','triangle':'20250106','shapely':'2.2.0','torch':'2.14.1+cu130','matplotlib':'3.10.0'}.items()}

def test_preloaded_supported_stack_is_retained_and_setup_twice_no_pip(monkeypatch):
    before=installed_profile();monkeypatch.setattr(bootstrap,'environment_versions',lambda:copy.deepcopy(before));calls=[]
    def forbidden(*a,**k):calls.append(a);raise AssertionError('attempted replacing existing binaries')
    for _ in range(2):
        r=bootstrap.prepare_reference_environment(ROOT,is_colab=True,runner=forbidden,progress=lambda s:None)
        assert not r['installed'] and not r['restart_required'] and r['qualification_required']
        assert r['versions']==before
    assert not calls

def test_contaminated_session_fails_before_any_mutation(monkeypatch):
    profile=installed_profile();profile['numpy']['installed']='2.5.4';monkeypatch.setattr(bootstrap,'environment_versions',lambda:profile)
    def forbidden(*a,**k):pytest.fail('unsafe pip installation')
    with pytest.raises(RuntimeError,match='Contaminated session.*numpy loaded=2.4.6, installed=2.5.4'):
        bootstrap.prepare_reference_environment(ROOT,is_colab=True,runner=forbidden,progress=lambda s:None)

def test_missing_geometry_package_installs_without_numpy_resolution(monkeypatch):
    profile=installed_profile();profile['triangle']={'installed':None,'loaded':None};monkeypatch.setattr(bootstrap,'environment_versions',lambda:profile);calls=[]
    def install(argv,**kw):calls.append(argv);profile['triangle']={'installed':'20250106','loaded':None};return SimpleNamespace(returncode=0)
    r=bootstrap.prepare_reference_environment(ROOT,is_colab=True,runner=install,progress=lambda s:None)
    assert r['packages_installed']==['triangle'] and '--no-deps' in calls[0]
    assert calls[0][-1]=='triangle==20250106' and all('numpy' not in x for x in calls[0])

def test_project_install_is_idempotent_and_does_not_import_science(tmp_path,monkeypatch):
    (tmp_path/'pyproject.toml').write_text('[project]\nname="fixture"\n');calls=[]
    def installer(argv,**kw):calls.append(argv);return SimpleNamespace(returncode=0)
    before={n:sys.modules.get(n) for n in ('numpy','scipy','torch')}
    bootstrap.install_project(tmp_path,runner=installer,progress=lambda s:None,source_revision='fixed')
    bootstrap.install_project(tmp_path,runner=installer,progress=lambda s:None,source_revision='fixed')
    assert len(calls)==1 and '--no-deps' in calls[0] and '--no-build-isolation' in calls[0]
    assert all(sys.modules.get(n) is module for n,module in before.items())

def test_install_interrupt_reaps_process(monkeypatch):
    original=subprocess.Popen;children=[]
    def spawn(*a,**kw):
        child=original(*a,**kw);children.append(child);communicate=child.communicate;first=[True]
        def interrupted(*args,**kwargs):
            if first[0]:first[0]=False;raise KeyboardInterrupt()
            return communicate(*args,**kwargs)
        child.communicate=interrupted;return child
    monkeypatch.setattr(subprocess,'Popen',spawn)
    with pytest.raises(KeyboardInterrupt):bootstrap.run_install([sys.executable,'-c','import time;time.sleep(60)'],progress=lambda s:None)
    assert len(children)==1 and children[0].poll() is not None

@pytest.mark.parametrize('family',GEOMETRIES)
def test_materialized_mesh_is_exact_original_certified_mesh(family,monkeypatch):
    parent=load_record(current=False)
    import triangle
    monkeypatch.setattr(triangle,'triangulate',lambda *a,**k:pytest.fail('certified geometry was regenerated'))
    mesh,spec,origin=resolve_reference(family);expected=parent['mesh_records'][family]['production']
    assert mesh.mesh_id==expected['mesh_id'] and spec.mesh_id==mesh.mesh_id
    assert spec.h==expected['h']==.02 and spec.degree==expected['degree']==4
    assert origin=='certified artifact' and not mesh.vertices.flags.writeable


def test_real_reference_change_has_diagnostic_fields_and_is_rejected():
    record=load_record();rows=report_selection(record,('square',),policy={**POLICY,'reference_h':.04},progress=lambda s:None)
    assert {'h','mesh_id'}<=set(rows[0]['differing_fields'])
    with pytest.raises(ValidationError,match='recertification|policy'):
        authorize(record,families=('square',),policy={**POLICY,'reference_h':.04})


def test_model_only_changes_reuse_same_reference_certificate(monkeypatch):
    from pdeep.smooth.qualification import ensure_compatible_certificate
    from pdeep.smooth.workflows import prepare
    record,path=ensure_compatible_certificate(progress=lambda s:None)
    def forbidden(*a,**kw):pytest.fail('model-only setting caused reference propagation')
    monkeypatch.setattr('pdeep.smooth.reference.Reference.dynamic',forbidden)
    ids=[]
    for models,capacity in [(('A_GEO',),'medium'),(('C_GEO',),'medium'),(('A_GEO','C_GEO'),'large')]:
        plan=prepare(models=models,capacity=capacity,experiment='tiny_overfit',updates=100,device='cpu',certificate_path=path)
        ids.append(plan.certificate['certification_id']);assert all(c.updates==100 for c in plan.configs)
    assert ids==[record['certification_id']]*3


def test_bootstrap_order_and_no_unsafe_reload_tricks():
    nb=json.loads((ROOT/'notebooks/02_train.ipynb').read_text());setup=''.join(nb['cells'][2]['source'])
    assert setup.index('prepare_reference_environment')<setup.index('install_project')<setup.index('from pdeep.colab_setup import finish_setup')
    assert 'install_checkout' not in setup
    src=(ROOT/'src/pdeep/smooth/bootstrap.py').read_text()
    assert 'importlib.reload' not in src and 'del sys.modules' not in src
    assert 'ensure_compatible_certificate' in ''.join(nb['cells'][7]['source'])
    assert 'PLAN = prepare' in ''.join(nb['cells'][8]['source'])


def test_changed_time_step_is_reported_as_a_real_numeric_difference():
    parent=load_record(current=False)
    row=report_selection(parent,('square',),policy={**POLICY,'reference_dt':.004},progress=lambda s:None)[0]
    assert row['differing_fields']['dt']=={'selected':.004,'certified':.002}


def test_actual_fresh_process_preloaded_numpy_then_setup_imports_twice():
    """Real subprocess, real imports, no pip/no Colab claim."""
    program=r'''
import sys,runpy,json
from pathlib import Path
import numpy
before=numpy
helper=runpy.run_path('src/pdeep/smooth/bootstrap.py')
def forbidden(*a,**k):raise AssertionError('existing dependencies were unexpectedly replaced')
for _ in range(2):
 result=helper['prepare_reference_environment'](Path('.'),is_colab=True,runner=forbidden,progress=lambda s:None)
 assert not result['installed'] and not result['restart_required']
import scipy,torch
from pdeep.smooth.certification import load_record,authorize
record=load_record();authorize(record)
assert numpy is before
print(json.dumps({'executable':sys.executable,'numpy':numpy.__version__,'scipy':scipy.__version__,'certificate':record['certification_id'],'same_loaded_numpy':True}))
'''
    r=subprocess.run([sys.executable,'-c',program],cwd=ROOT,text=True,capture_output=True,timeout=45)
    assert r.returncode==0,r.stderr
    assert json.loads(r.stdout)['same_loaded_numpy']
