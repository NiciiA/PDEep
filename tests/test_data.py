from dataclasses import fields, replace
import json
import numpy as np
import pytest
import torch

from pdeep.batches import QueryBatch
from pdeep.core import ValidationError
from pdeep.data import (SPLITS, SENSOR_COUNTS, CaseDataset, CaseKey, Normalizers, calibrate,
                        observations, reload_observations, training_queries, collate_inputs,
                        case_coefficients, mixture_counts, evaluation_grid, EVALUATION_TIMES)
from pdeep.reference import canonical_spec, evaluate_batch


@pytest.fixture(scope="module")
def normalizers():
    return calibrate(cases=64,points_per_case=64)


def test_disjoint_case_ids_and_lazy_reproducibility():
    d=CaseDataset()
    assert iter(d.cases("train","wave")) is not None
    for p in SENSOR_COUNTS:
        sets=[{CaseKey(s,p,i).physical_id for i in range(n)} for s,n,_ in SPLITS]
        assert [len(s) for s in sets]==[12000,1024,1024]
        assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        a=d.case("train",p,0)
        assert a==d.case("train",p,0)
        assert a.identity!=d.case("validation",p,0).identity
        assert a.xi!=d.case("test",p,0).xi
        assert a.xi!=CaseDataset(1).case("train",p,0).xi
    with pytest.raises(ValidationError):
        d.case("validation","heat",1024)
    assert json.dumps(d.summary())


def test_normalizer_provenance_persistence_and_reproducibility(normalizers,tmp_path,monkeypatch):
    # Calibration has no dataset argument and cannot read any split's cases.
    monkeypatch.setattr(CaseDataset,"case",lambda *args: (_ for _ in ()).throw(AssertionError("case access")))
    again=calibrate(cases=64,points_per_case=64)
    assert again==normalizers
    assert again.distribution=="train_prior_and_training_query_risk"
    assert calibrate(root_seed=1,cases=8,points_per_case=16).identity!=again.identity
    path=again.save(tmp_path)
    assert Normalizers.load(path)==again
    assert again.save(tmp_path)==path
    assert len(again.coefficient_scales[2][1])==18
    raw=json.loads(path.read_text()); raw['quantities'][0]['rms']*=2
    path.write_text(json.dumps(raw))
    with pytest.raises(ValidationError,match="checksum"):
        Normalizers.load(path)


@pytest.mark.parametrize("pde",list(SENSOR_COUNTS))
def test_observations_and_queries(pde,normalizers):
    d=CaseDataset(); c=d.case("train",pde,0)
    obs=observations(d,c,normalizers)
    n=SENSOR_COUNTS[pde]
    assert obs.inputs.position.shape==(1,n,2)
    assert obs.inputs.value.shape==obs.metadata.clean.shape==(1,n,1)
    assert obs.inputs.valid.all()
    assert (obs.inputs.position>.05).all() and (obs.inputs.position<.95).all()
    assert torch.equal(obs.inputs.value,obs.metadata.clean+obs.metadata.noise)
    assert (obs.metadata.eta>=.02).all() and (obs.metadata.eta<=.08).all()
    assert not any(f.name in {"clean","noise","eta"} for f in fields(obs.inputs))
    for q in torch.unique(obs.inputs.quantity_id):
        mask=obs.inputs.quantity_id==q
        torch.testing.assert_close(obs.inputs.sigma[mask],obs.metadata.eta[mask]*normalizers.scale(pde,int(q)),rtol=0,atol=0)
    repeat=observations(d,c,normalizers)
    assert repeat.metadata.identity==obs.metadata.identity
    assert torch.equal(repeat.inputs.value,obs.inputs.value)
    other=observations(d,c,normalizers,set_id="heldout")
    assert other.metadata.identity!=obs.metadata.identity
    assert not torch.equal(other.inputs.position,obs.inputs.position)
    if pde=="poisson":
        assert not obs.inputs.time_present.any()
        assert (obs.inputs.physical_time==0).all()
    else:
        t=obs.inputs.physical_time.flatten()
        assert obs.inputs.time_present.all()
        assert int((t==0).sum())==obs.metadata.time_counts[0]>=1
        assert int(((t>0)&(t<.1)).sum())==obs.metadata.time_counts[1]>=1
        assert int((t>=.1).sum())==obs.metadata.time_counts[2]
    if pde=="wave":
        assert torch.bincount(obs.inputs.quantity_id.flatten()).tolist()==[32,32]
    query=training_queries(d,c)
    assert query.inputs.position.shape==(1,128,2) and query.target.shape==(1,128,1)
    assert query.identity==training_queries(d,c).identity
    if pde!="poisson":
        assert query.time_counts==(16,16,96)
    obs.inputs.validate((canonical_spec(pde),))


def test_rng_isolation_placement_not_truth_and_global_states(normalizers):
    d=CaseDataset(); c=d.case("train","wave",3)
    np_before=np.random.get_state(); torch_before=torch.random.get_rng_state().clone()
    first=observations(d,c,normalizers)
    d.streams.numpy("monitoring", "many").normal(size=1000)
    d.case("test","wave",42); training_queries(d,c,count=32)
    again=observations(d,c,normalizers)
    assert torch.equal(first.inputs.value,again.inputs.value)
    altered=replace(c,xi=tuple(2*x for x in c.xi))
    changed=observations(d,altered,normalizers)
    for field in ("position","physical_time","quantity_id","sigma"):
        assert torch.equal(getattr(first.inputs,field),getattr(changed.inputs,field))
    assert not torch.equal(first.metadata.clean,changed.metadata.clean)
    assert torch.equal(first.metadata.noise,changed.metadata.noise)
    assert torch.equal(torch_before,torch.random.get_rng_state())
    assert np.array_equal(np_before[1],np.random.get_state()[1])
    assert np_before[2:]==np.random.get_state()[2:]


def test_measurement_noise_distribution_and_independence(normalizers):
    d=CaseDataset(); c=d.case("train","wave",8)
    obs=observations(d,c,normalizers,count=16384)
    z=(obs.metadata.noise/obs.inputs.sigma).flatten().numpy()
    assert abs(z.mean())<.04 and abs(z.std()-1)<.04
    assert abs(np.corrcoef(z[:-1],z[1:])[0,1])<.04
    assert obs.inputs.sigma.std()>0


def test_observation_persistence_and_masked_collation(normalizers,tmp_path):
    d=CaseDataset()
    cases=tuple(d.case("validation",p,7) for p in SENSOR_COUNTS)
    records=[observations(d,c,normalizers) for c in cases]
    for record in records:
        path=record.save(tmp_path)
        assert path==record.save(tmp_path)
        loaded=reload_observations(path,d,normalizers)
        assert torch.equal(loaded.inputs.value,record.inputs.value)
        assert loaded.metadata.identity==record.metadata.identity
    packed=collate_inputs([r.inputs for r in records])
    specs=tuple(canonical_spec(c.key.pde) for c in cases)
    packed.validate(specs)
    assert packed.valid.sum(1).tolist()==[32,48,64]
    pred=evaluate_batch(specs,case_coefficients(cases),packed)
    for i,r in enumerate(records):
        torch.testing.assert_close(pred[i,:r.inputs.position.shape[1]],r.metadata.clean[0],rtol=1e-13,atol=1e-13)
    assert torch.equal(pred[~packed.valid],torch.zeros_like(pred[~packed.valid]))
    path=records[0].save(tmp_path)
    content=json.loads(path.read_text());content['count']+=1
    path.write_text(json.dumps(content))
    with pytest.raises(ValidationError,match="checksum"):
        reload_observations(path,d,normalizers)


def test_counts_and_evaluation_grid():
    for n in (2,3,8,16,32,48,64,128):
        counts=mixture_counts(n,(.1,.3,.6))
        assert sum(counts)==n and min(counts[:2])>=1
    q=evaluation_grid("wave",quantity_id=1)
    assert q.position.shape==(1,9*64*64,2)
    assert tuple(torch.unique(q.physical_time).tolist())==EVALUATION_TIMES
    with pytest.raises(ValidationError):
        observations(CaseDataset(),CaseDataset(1).case("train","heat",0),calibrate(cases=1,points_per_case=8))


def test_fresh_process_observation_reconstruction(normalizers,tmp_path):
    import subprocess
    import sys
    d=CaseDataset()
    record=observations(d,d.case("test","wave",17),normalizers)
    normalizer_path=normalizers.save(tmp_path/"normalizers")
    record_path=record.save(tmp_path/"observations")
    program="""from pdeep.data import CaseDataset, Normalizers, reload_observations
import sys
n=Normalizers.load(sys.argv[1])
o=reload_observations(sys.argv[2],CaseDataset(n.root_seed),n)
print(o.metadata.identity)
"""
    result=subprocess.run([sys.executable,"-c",program,str(normalizer_path),str(record_path)],capture_output=True,text=True,check=True)
    assert result.stdout.strip()==record.metadata.identity
