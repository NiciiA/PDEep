from dataclasses import replace
import hashlib
import inspect
import pytest
import torch

from pdeep.core import ValidationError
from pdeep.geometry import BoxGeometry,AnchorLayout
from pdeep.encoding import CoordinateFeatures,EquationVocabulary
from pdeep.memory import SpatialMemory,gaussian_readout,MemoryRetriever
from pdeep.data import calibrate
from pdeep.reference import canonical_spec
from pdeep.representations import build_representations
from pdeep.retrieval_checks import kernel_checks,retrieval_checks


def test_gaussian_formula_continuity_and_derivatives():
    report=kernel_checks()
    assert report["passed"],[c for c in report["checks"] if not c["passed"]]
    assert [s["anchors"] for s in report["scales"]]==[16,64]
    assert all(s["gradcheck"] for s in report["scales"])


@pytest.mark.parametrize("device",["cpu","mps","cuda"])
def test_full_retrieval_continuity_derivatives_and_batches(device):
    from pdeep.runtime import capabilities
    if not next(c.available for c in capabilities() if c.device==device):
        pytest.skip(device+" unavailable")
    report=retrieval_checks(device=device)
    assert report["passed"] and report["weights_shape"][-1]==80


def test_geometry_provider_owns_placement_boundary_and_scale_membership():
    class CustomGeometry:
        boundary_feature_size=4
        def anchor_layout(self):
            base=BoxGeometry().anchor_layout()
            # Reverse level order; generic layers cannot assume coarse comes first.
            order=torch.arange(79,-1,-1)
            return AnchorLayout(.8*base.positions[order]+.1,base.levels[order])
        def boundary_features(self,p):
            return torch.cat((2*p,3-p),dim=-1)
    custom=CustomGeometry();memory=SpatialMemory(geometry_provider=custom)
    layout=custom.anchor_layout()
    assert torch.equal(memory.positions,layout.positions) and torch.equal(memory.levels,layout.levels)
    assert not torch.equal(memory.positions,BoxGeometry().anchor_layout().positions)
    features=CoordinateFeatures(geometry_provider=custom)
    p=torch.tensor([[[.2,.3]]]);t=torch.zeros(1,1,1);mask=torch.ones(1,1,dtype=torch.bool)
    encoded=features(p,t,~mask,mask)
    torch.testing.assert_close(encoded[...,2:6],custom.boundary_features(p),rtol=0,atol=0)
    # Layouts are geometry-owned even outside the unit square; this does not
    # extend the unchanged physical ProblemSpec/domain validator.
    rectangular=BoxGeometry(bounds=((-2.,3.),(4.,7.))).anchor_layout()
    assert rectangular.positions[:,0].min()>-2 and rectangular.positions[:,1].min()>4
    assert rectangular.positions[:,0].max()<3 and rectangular.positions[:,1].max()<7
    assert 'topk' not in inspect.getsource(MemoryRetriever)
    from pdeep.encoding import ObservationEncoding
    encoded_obs=ObservationEncoding(torch.zeros(1,2,96),torch.zeros(1,96),torch.tensor([[[.3,.7],[.7,.3]]]),
                                    torch.zeros(1,2,1),torch.zeros(1,2,dtype=torch.bool),torch.zeros(1,2,dtype=torch.int64),
                                    torch.zeros(1,2,1),torch.ones(1,2,dtype=torch.bool))
    state=memory(encoded_obs,torch.zeros(1,96))
    assert torch.equal(state.positions,layout.positions) and torch.isfinite(state.tokens).all()
    # Same parameters, permuted storage: level-aware mixing must commute with permutation.
    default_memory=SpatialMemory()
    default_memory.load_state_dict({k:v for k,v in memory.state_dict().items() if k not in {'positions','levels'}},strict=False)
    default_memory.positions.copy_(memory.positions.flip(0));default_memory.levels.copy_(memory.levels.flip(0))
    # Match the provider too; this test isolates storage ordering rather than boundary values.
    default_memory.domain_geometry=custom
    normal=default_memory(encoded_obs,torch.zeros(1,96))
    torch.testing.assert_close(normal.tokens.flip(1),state.tokens,rtol=1e-5,atol=2e-6)


def test_masked_query_gradients_and_memory_gradients():
    from pdeep.batches import QueryBatch
    from pdeep.memory import MemoryState
    spec=canonical_spec("poisson");layout=BoxGeometry().anchor_layout()
    p=torch.tensor([[[.5,.5],[float("nan"),float("nan")]]],requires_grad=True)
    valid=torch.tensor([[True,False]])
    query=QueryBatch(p,torch.full((1,2,1),float("nan")),torch.zeros_like(valid),
                     torch.tensor([[0,-999]]),valid)
    tokens=torch.sin(torch.arange(80*96,dtype=torch.float32)).reshape(1,80,96).requires_grad_()
    state=MemoryState(tokens,torch.zeros(1,96),layout.positions,layout.levels,None)
    features=torch.zeros(1,2,96);features[:,1]=float("nan")
    module=MemoryRetriever();out=module(state,query,features,(spec,))
    assert (out.features[:,1]==0).all() and (out.weights[:,1]==0).all()
    assert (out.coarse_context[:,1]==0).all() and (out.fine_context[:,1]==0).all()
    out.features[0,0,0].backward()
    assert torch.isfinite(p.grad).all() and (p.grad[:,1]==0).all()
    assert torch.isfinite(tokens.grad).all() and (tokens.grad.abs().sum(-1)>0).all()


@pytest.mark.parametrize("ell",[0.,-1.,float("nan"),float("inf")])
def test_invalid_bandwidth_fails(ell):
    with pytest.raises(ValidationError):MemoryRetriever(coarse_lengthscale=ell)


def test_default_geometry_and_C_coordinate_math_unchanged():
    from pdeep.encoding import FourierFeatures
    p=torch.tensor([[[.0,.0],[.3,.7],[1.,1.]]],requires_grad=True)
    t=torch.tensor([[[0.],[.2],[.9]]],requires_grad=True)
    valid=torch.ones(1,3,dtype=torch.bool);present=torch.tensor([[False,True,True]])
    safe_time=torch.where(present[...,None],t,torch.zeros_like(t))
    time_fourier=torch.where(present[...,None],FourierFeatures()(safe_time),torch.zeros(1,3,12))
    old_features=torch.cat((p,p,1-p,FourierFeatures()(p),safe_time,time_fourier,present[...,None].float()),dim=-1)
    new_features=CoordinateFeatures()(p,t,present,valid)
    torch.testing.assert_close(old_features,new_features,rtol=0,atol=0)
    projection=torch.arange(44,dtype=torch.float32)
    old_derivative=torch.autograd.grad((old_features*projection).sum(),(p,t),retain_graph=True)
    new_derivative=torch.autograd.grad((new_features*projection).sum(),(p,t))
    for a,b in zip(old_derivative,new_derivative):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_scale_readout_not_content_weighted_and_anchor_storage_invariant():
    layout=BoxGeometry().anchor_layout();positions=layout.positions
    tokens=torch.stack((positions[:,0],positions[:,1]),dim=-1)[None]
    p=torch.tensor([[[.5,.5],[.25,.625]]])
    m,w=gaussian_readout(p,positions,tokens,.3)
    # Changing memory values changes m, but Gaussian weights depend only on geometry.
    changed,cw=gaussian_readout(p,positions,2*tokens,.3)
    torch.testing.assert_close(cw,w,rtol=0,atol=0)
    torch.testing.assert_close(changed,2*m,rtol=0,atol=0)
    order=torch.arange(79,-1,-1)
    permuted,pw=gaussian_readout(p,positions[order],tokens[:,order],.3)
    torch.testing.assert_close(permuted,m,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(pw,w[...,order],rtol=1e-6,atol=1e-6)
