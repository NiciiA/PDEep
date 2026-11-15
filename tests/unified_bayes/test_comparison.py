import copy
import numpy as np
import pytest
from pdeep.unified_bayes.comparison import summarize,METHODS,QUANTITIES
from pdeep.unified_bayes.workflow import classical
from .conftest import EVIDENCE


def rows():
    out=[]
    for m in METHODS:
        for n in (8,16,32,64,128):
            for p in ('poisson','heat','wave'):
                for g in ('square','rectangle','lshape','disk','ellipse','hole','annulus','polygon'):
                    for case in range(2):
                        for q in ('u','v') if p=='wave' else ('u',):
                            out.append({'method':m,'case_id':f'{p}-{g}-{case}','split':'test','sensor_count':n,'pde':p,'family':g,'quantity':q,'observation_identity':f'{p}-{g}-{case}-{n}','query_identity':f'{p}-{g}-{case}-query','standardized_MSE':float(case+1),'physical_RMSE':.01,'relative_L2':.2,'cost':{'inference_seconds':.001}})
    return out


def test_all_six_tables_fairness_and_partial_status():
    r=summarize(rows(),replicates=100);assert len([v for v in r['tables'] if v['group']=='task'])==24*6
    assert len([v for v in r['tables'] if v['group']=='sensor_count'])==5*6
    assert all(v['poisson/u/standardized_MSE']==1.5 for v in r['tables'] if v['group']=='overall')
    assert all(v['mean_standardized_MSE_difference']==0 and v['paired_cases']==16 for v in r['paired_intervals'])
    partial=summarize([v for v in rows() if v['method'] in METHODS[:3]],replicates=10)
    assert len([v for v in partial['tables'] if v['group']=='overall'])==6 and all(v['status'].startswith('unavailable') for v in partial['tables'] if v['group']=='overall' and v['method'] in METHODS[3:])
    bad=rows();bad[0]['observation_identity']='wrong'
    with pytest.raises(ValueError,match='unmatched'):summarize(bad)
    incomplete=rows()[1:]
    with pytest.raises(ValueError,match='incomplete matched cohort'):summarize(incomplete)
    bad=rows();bad[0]['split']='validation'
    with pytest.raises(ValueError,match='TEST'):summarize(bad)
    EVIDENCE['matched_table_control']={'methods':6,'tasks':24,'counts':5,'physical_cases_per_task':2,'optimizer_updates':0,'metrics':'synthetic arithmetic only; no accuracy claim'}


def test_preview_no_reference_or_models(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('expensive constructor in preview')
    monkeypatch.setattr('pdeep.unified_bayes.workflow.ForwardOperators',forbidden)
    r=classical(None,'/missing',action='preview');assert r['reference_solves']==0 and r['posterior_evaluations']==0
