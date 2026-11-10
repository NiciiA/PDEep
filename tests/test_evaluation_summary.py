"""All-case reductions and bounded representative plotting/display."""
import json
from pathlib import Path
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
from pdeep.evaluation_summary import representative_cases,physics_summary_table,coverage_summary_table,refinement_summary_table
from pdeep.workflow_plotting import plot_evaluation
from pdeep.core import ValidationError


def fixture_result(tmp_path,count=50):
    times=[0,.02,.05,.1,.2,.35,.5,.75,1];cases=[]
    for pde in ('poisson','heat','wave'):
        qs=['u','v'] if pde=='wave' else ['u'];n=4096*(1 if pde=='poisson' else 9)
        for i in range(count):
            row={'pde':pde,'case_id':f'validation/{pde}/{i:05d}','split':'validation','training_status':'untrained',
                 'quantities':{q:{} for q in qs},
                 'time_errors':{q:[{'physical_time':t,'physical_MSE':float((i+1)**2)} for t in times] for q in qs},
                 'physics':{'units':['unit1','unit2'] if pde=='wave' else ['unit1'],
                            'physical_RMSE':[float(i+1)]*(2 if pde=='wave' else 1),
                            'normalized_RMSE':[float(i+1)/2]*(2 if pde=='wave' else 1),'boundary_max_abs':{'u':i*1e-9}},
                 'refinement':{'Phi_by_iteration':[20.,10.,5.,2.],
                     'coefficient_vs_truth':[{'physical_RMSE':float(i+1)}]*4,
                     'coefficient_vs_D_mean':[{'physical_RMSE':float(i+1)/2}]*4},
                 'uncertainty':{key:{'0.5':.4,'0.9':.8,'0.95':.9} for key in ('coefficient_coverage','heldout_latent_coverage','heldout_measurement_coverage')}}
            row['uncertainty']['field_quantities']={q:{'latent_coverage':{'0.5':.5,'0.9':.9,'0.95':.95}} for q in qs}
            cases.append(row)
        # Only the representative's field assets exist. Plotting all 150 rows
        # would fail, as well as needlessly regenerating duplicate figures.
        asset={'truth_'+q:np.zeros((1,n,1)) for q in qs}
        asset.update({'prediction_'+q:np.zeros((1,n,1)) for q in qs})
        asset.update({'latent_variance_'+q:np.ones((1,n,1))*.01 for q in qs})
        asset.update(conditioning_quantity=np.zeros((1,2)),conditioning_position=np.array([[[.2,.3],[.6,.7]]]))
        for name in ('D','B0'):np.savez_compressed(tmp_path/f'{name}__validation_{pde}_00000.npz',**asset)
    return {'directory':str(tmp_path),'scientific':{'cases':{'D':cases,'B0':cases},
        'methods':{n:{'training_status':'untrained' if n=='B0' else 'exact; no training','training_selection':{'successful_updates':0},'checkpoint':n+'.json'} for n in ('D','B0')},
        'aggregate':{'B0':{'by_PDE':{p:{} for p in ('poisson','heat','wave')}}}}}


def test_tables_aggregate_every_case_and_preserve_components(tmp_path):
    r=fixture_result(tmp_path,count=2)
    physics=physics_summary_table(r)
    assert physics.count('| wave | 2 |')==4 # two equation components x two methods
    assert '| 1.5811 |' in physics # sqrt((1^2+2^2)/2), not mean RMSE=1.5
    assert 'unit2' in physics and '| 1e-09 |' in physics
    coverage=coverage_summary_table(r)
    assert '| wave | 2 | field v latent | 0.5000 | 0.9000 | 0.9500 |' in coverage
    refinement=refinement_summary_table(r)
    assert '| B0 | poisson | 2 | 3 | 2 | 1.5811 | 0.79057 |' in refinement
    assert representative_cases(r['scientific'])['wave']['case_id'].endswith('00000')
    assert representative_cases(r['scientific'],{'wave':1})['wave']['case_id'].endswith('00001')
    with pytest.raises(ValidationError):representative_cases(r['scientific'],{'wave':99})


def test_150_cases_generate_only_unique_bounded_representative_plots(tmp_path):
    r=fixture_result(tmp_path)
    paths=plot_evaluation(r)
    assert len(paths)==10 and len(set(paths))==10
    assert sum(p.name.startswith('D_uncertainty_') for p in paths)==3
    assert all(p.is_file() and p.parent==tmp_path/'summary_plots' for p in paths)
    assert not __import__('matplotlib.pyplot',fromlist=['get_fignums']).get_fignums()


def test_notebook_keeps_user_model_selection_and_summarizes_instead_of_printing_cases():
    root=Path(__file__).resolve().parents[1];n=json.loads((root/'notebooks/03_evaluate.ipynb').read_text())
    assert 'physics_summary_table(RESULT)' in ''.join(n['cells'][9]['source'])
    assert 'for row in rows' not in ''.join(n['cells'][9]['source'])
    assert 'coverage_summary_table(RESULT)' in ''.join(n['cells'][10]['source'])
    assert 'representative_case_ids=REPRESENTATIVE_CASE_IDS' in ''.join(n['cells'][7]['source'])
