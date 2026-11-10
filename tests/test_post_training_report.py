"""Saved benchmark display must stay bounded and reject mismatched provenance."""
from types import SimpleNamespace
import json
from pathlib import Path
import pytest
from pdeep.core import ValidationError
from pdeep.post_training_report import load_for_display,compact_table,FOCUS


def test_saved_display_rejects_changed_checkpoint_cases_or_normalizers(tmp_path,monkeypatch):
    p=tmp_path/'result.json'
    r={'version':'post-training-optimization-v1','neural_training_updates':0,'checkpoint_artifact_id':'id',
       'normalizer_identity':'norm','cases':[{'case_identity':'case','conditioning_identity':'obs'}]}
    p.write_text(json.dumps(r));plan=SimpleNamespace(normalizer_identity='norm',cases=[SimpleNamespace(case_identity='case',conditioning_identity='obs')])
    monkeypatch.setattr('pdeep.post_training_report.read_manifest',lambda *a,**kw:{'artifact_id':'id'})
    assert load_for_display(p,plan,'unused')==r
    plan.cases[0].conditioning_identity='changed'
    with pytest.raises(ValidationError,match='observation plan mismatch'):load_for_display(p,plan,'unused')
    plan.cases[0].conditioning_identity='obs';plan.normalizer_identity='different'
    with pytest.raises(ValidationError,match='normalizers'):load_for_display(p,plan,'unused')
    plan.normalizer_identity='norm';monkeypatch.setattr('pdeep.post_training_report.read_manifest',lambda *a,**kw:{'artifact_id':'other'})
    with pytest.raises(ValidationError,match='checkpoint'):load_for_display(p,plan,'unused')


def test_compact_table_has_separate_hybrid_names_and_per_PDE_costs():
    m={k:{'metrics':{n:{'mean':.01} for n in ('coefficient_standardized_MSE','field_standardized_MSE','optimality_gap')},
          'iterations':{'median':3},'timing':{'coefficient_total':{'median':.002,'p95':.003},'classical_tail':{'median':.0001}}} for k in FOCUS}
    text=compact_table({'summary':{p:m for p in ('poisson','heat','wave')}})
    assert len(text.splitlines())==20
    assert text.count('B0+Classical CG')==3 and '2.000 / 3.000' in text and '0.100' in text


def test_notebook_only_reads_benchmark_and_does_not_run_it():
    root=Path(__file__).resolve().parents[1];n=json.loads((root/'notebooks/03_evaluate.ipynb').read_text())
    code='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
    assert 'load_for_display(' in code and 'compact_table(' in code
    assert 'run_benchmark(' not in code and 'benchmark_post_training.py' not in code
