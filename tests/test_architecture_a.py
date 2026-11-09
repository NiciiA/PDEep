from dataclasses import replace
import json
import subprocess
import sys
import pytest
import torch
from pdeep.a_checks import mixed_fixture,architecture_a_checks,slice_tokens
from pdeep.a_training import objective,ATrainingConfig,train_update,make_training_batch
from pdeep.architecture_a import build_architecture_a
from pdeep.a_artifacts import save_inference,load_inference
from pdeep.batches import UnknownCoefficients
from pdeep.core import ValidationError
from pdeep.data import calibrate,CaseDataset
from pdeep.geometry import BoxGeometry
from pdeep.representations import build_representations
from pdeep.models import select_architecture


@pytest.fixture(scope="module")
def norm():return calibrate()


@pytest.mark.parametrize("device",["cpu","mps","cuda"])
def test_A1_through_A10(norm,device):
    from pdeep.runtime import capabilities
    if not next(c.available for c in capabilities() if c.device==device):pytest.skip(device+" unavailable")
    r=architecture_a_checks(norm,device=device)
    assert r["passed"],[x for x in r["checks"] if not x["passed"]]
    assert all(not g["missing"] and not g["nonfinite"] for g in r["gradients"].values())


def test_objective_independent_case_quantity_reduction_and_padding(norm):
    a=build_architecture_a(norm);b=mixed_fixture(norm,queries=10)
    out=a(b.specs,b.observations,b.queries)
    # Deliberately imbalanced quantity counts: wave 9 u, 1 v. Equal q weighting.
    ids=b.queries.quantity_id.clone();ids[2]=0;ids[2,-1]=1
    valid=b.queries.valid.clone();valid[0,-2:]=False
    q=replace(b.queries,quantity_id=ids,valid=valid);b=replace(b,queries=q)
    scales=out.context.quantity_scales.gather(1,ids)[...,None]
    residual=torch.ones_like(out.predicted_value)*2
    residual[2,-1]=4
    prediction=b.target+scales*residual
    prediction[~valid]=float("nan")
    delta=out.context.prior_std*3
    xi=b.unknowns.xi+delta;xi[~out.xi_active]=float("nan")
    fake=replace(out,predicted_value=prediction,xi_hat=xi)
    l=objective(fake,b)
    # P1=4, heat=4, wave=(4+16)/2=10 -> 6; all xi=9.
    assert float(l["field"])==pytest.approx(6.,abs=2e-6)
    assert float(l["xi"])==pytest.approx(9.,abs=2e-6)
    assert float(l["total"])==pytest.approx(6.9,abs=2e-6)
    assert float(objective(fake,b,coefficient_weight=0)["total"])==pytest.approx(6.)
    # All-empty queries contribute zero field; no padding mean dilution.
    empty=replace(b,queries=replace(q,valid=torch.zeros_like(valid)))
    assert float(objective(fake,empty)["field"])==0.


def test_microbatch_gradient_accumulation_uneven_final(norm):
    a=build_architecture_a(norm);b=mixed_fixture(norm,queries=16)
    out=a(b.specs,b.observations,b.queries);objective(out,b)["total"].backward()
    expected={n:p.grad.clone() for n,p in a.named_parameters()}
    a.zero_grad(set_to_none=True)
    for lo,hi in ((0,2),(2,3)):
        part=b.slice(lo,hi);out=a(part.specs,part.observations,part.queries)
        (objective(out,part)["total"]*((hi-lo)/3)).backward()
    for n,p in a.named_parameters():torch.testing.assert_close(p.grad,expected[n],rtol=3e-4,atol=3e-7)


def test_geometry_neighbors_no_wrap_and_boundary_interface(norm):
    g=BoxGeometry();w=g.neighbor_weights();p=g.anchor_layout().positions
    assert w.shape==(80,80) and (w.diag()==0).all()
    fine=w[16:,16:]>0
    delta=(p[16:,None]-p[None,16:]).abs()
    assert (delta[fine].sum(-1)-.125).abs().max()<1e-6
    assert (w[:16,16:]==0).all() and (w[16:,:16]>0).sum(1).eq(4).all()
    torch.testing.assert_close(g.boundary_distance(torch.tensor([[.2,.3],[.9,.6]])),torch.tensor([[.2],[.1]]))
    with pytest.raises(ValidationError,match="unit square"):build_architecture_a(norm,geometry=BoxGeometry(bounds=((0.,2.),(0.,1.))))
    b=mixed_fixture(norm)
    with pytest.raises(ValidationError):
        bad=replace(b.specs[0],geometry=replace(b.specs[0].geometry,kind="irregular"))
        build_architecture_a(norm).prepare_context((bad,*b.specs[1:]),b.observations)


def test_no_solver_path_or_target_leakage_and_independent_owners(norm,monkeypatch):
    b=mixed_fixture(norm);before=torch.random.get_rng_state().clone();a=build_architecture_a(norm)
    assert torch.equal(before,torch.random.get_rng_state())
    same=build_architecture_a(norm)
    assert all(torch.equal(p,q) for p,q in zip(a.parameters(),same.parameters()))
    owners=[build_representations(owner,norm,a.vocabulary) for owner in ("B","C")]
    pointers=[{p.data_ptr() for p in m.parameters()} for m in (a,*owners)]
    assert not(pointers[0]&pointers[1] or pointers[0]&pointers[2] or pointers[1]&pointers[2])
    assert all(m.memory is None for m in owners)
    def forbidden(*args,**kwargs):raise AssertionError("solver called in A")
    import pdeep.reference as reference
    import pdeep.bayes as bayes
    for name in ("evaluate","evaluate_batch","basis"):monkeypatch.setattr(reference,name,forbidden)
    monkeypatch.setattr(bayes,"sensor_matrix",forbidden)
    with torch.no_grad():
        out=a(b.specs,b.observations,b.queries)
        assert not hasattr(out.context,"xi_true") and not hasattr(out.context,"target")
        # Auxiliary head cannot affect the predicted field.
        a.coefficient_head[-1].bias.add_(20.)
        after=a.predict_queries(out.context,b.queries)
        torch.testing.assert_close(out.predicted_value,after.predicted_value,rtol=0,atol=0)
    with pytest.raises(ValidationError,match="different model"):same.predict_queries(out.context,b.queries)


def test_inference_reload_fresh_process_and_checksum(norm,tmp_path):
    a=build_architecture_a(norm);b=mixed_fixture(norm,queries=8)
    with torch.no_grad():expected=a(b.specs,b.observations,b.queries).predicted_value
    path=save_inference(a,tmp_path,source="fixture",config={},dataset_id=CaseDataset().identity)
    assert save_inference(a,tmp_path,source="fixture",config={},dataset_id=CaseDataset().identity)==path
    loaded,meta=load_inference(path)
    with torch.no_grad():actual=loaded(b.specs,b.observations,b.queries).predicted_value
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    # Fresh interpreter reconstructs normalizers/vocabulary/model with weights-only loading.
    code='from pdeep.a_artifacts import load_inference; import sys; a,m=load_inference(sys.argv[1]); print(sum(p.numel() for p in a.parameters()))'
    r=subprocess.run([sys.executable,"-c",code,str(path)],capture_output=True,text=True,check=True)
    assert int(r.stdout.strip())==sum(p.numel() for p in a.parameters())
    weights=path.parent/"weights.pt";weights.write_bytes(weights.read_bytes()+b"corrupt")
    with pytest.raises(ValidationError,match="checksum"):load_inference(path)


def test_actual_training_sampler_uniform_prior_no_test_and_default_budget(norm):
    batch=make_training_batch(CaseDataset(),norm,0,count=16)
    again=make_training_batch(CaseDataset(),norm,0,count=16)
    assert batch.case_ids==again.case_ids and torch.equal(batch.target,again.target)
    assert all("test" not in name for name in batch.case_ids)
    assert set(s.name for s in batch.specs)=={"poisson","heat","wave"}
    for row,spec in enumerate(batch.specs):
        if spec.name=="wave":assert batch.queries.quantity_id[row].sum()==64
    assert ATrainingConfig().updates==3 and not ATrainingConfig().tiny_overfit
    for kwargs in ({"updates":6},{"preset":"research"},{"preset":"tiny_overfit","updates":30}):
        with pytest.raises(ValidationError):ATrainingConfig(**kwargs)
    assert select_architecture("D").trainable_parameters==0
    assert type(select_architecture("A",normalizers=norm)) is type(build_architecture_a(norm))
    assert select_architecture("C",normalizers=norm).__class__.__name__=="ArchitectureC"
    for name in ("B","T2"):
        with pytest.raises(ValidationError,match="unsupported"):select_architecture(name)
