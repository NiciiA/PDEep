from dataclasses import fields,replace
import pytest
import torch

from pdeep.batches import QueryBatch
from pdeep.core import Status,TensorType,ValidationError
from pdeep.data import CaseDataset,calibrate,observations,training_queries,collate_inputs,case_coefficients
from pdeep.encoding import (CoordinateFeatures,EquationVocabulary,NUMERIC_NAMES,pack_graphs)
from pdeep.graph import GraphBuilder,Op
from pdeep.memory import coefficient_inputs,anchor_geometry
from pdeep.reference import canonical_spec
from pdeep.representations import build_representations


def float_batch(batch,device="cpu"):
    return type(batch)(**{f.name:getattr(batch,f.name).to(device=device,dtype=torch.float32 if getattr(batch,f.name).is_floating_point() else getattr(batch,f.name).dtype) for f in fields(batch)})


def permute(batch,index):
    return type(batch)(**{f.name:getattr(batch,f.name)[:,index] for f in fields(batch)})


def pad(batch,count):
    values={}
    for field in fields(batch):
        t=getattr(batch,field.name)
        fill=False if t.dtype==torch.bool else (-999 if t.dtype==torch.int64 else float("nan"))
        extra=torch.full((len(t),count,*t.shape[2:]),fill,dtype=t.dtype,device=t.device)
        values[field.name]=torch.cat((t,extra),dim=1)
    return type(batch)(**values)


@pytest.fixture(scope="module")
def fixture():
    specs=tuple(canonical_spec(p) for p in ("poisson","heat","wave"))
    dataset=CaseDataset();norm=calibrate()
    cases=tuple(dataset.case("train",s.name,0) for s in specs)
    obs=float_batch(collate_inputs([observations(dataset,c,norm).inputs for c in cases]))
    query=float_batch(collate_inputs([training_queries(dataset,c,count=32).inputs for c in cases]))
    return specs,norm,cases,obs,query,EquationVocabulary.from_specs(specs)


def test_graph_shapes_padding_values_and_gradients(fixture):
    specs,norm,_,_,_,vocab=fixture
    module=build_representations("C",norm,vocab).equation
    packed=pack_graphs(specs,vocab)
    encoding=module(packed)
    assert encoding.nodes.shape==(3,15,96) and encoding.context.shape==(3,96)
    padded=pack_graphs(specs,vocab,pad_nodes=5,pad_edges=7)
    padded=replace(padded,numeric=torch.where(padded.valid[...,None],padded.numeric,torch.full_like(padded.numeric,float("nan"))))
    after=module(padded)
    torch.testing.assert_close(encoding.nodes,after.nodes[:,:15],rtol=1e-5,atol=2e-6)
    torch.testing.assert_close(encoding.context,after.context,rtol=1e-5,atol=2e-6)
    assert (after.nodes[~padded.valid]==0).all()
    heat=specs[1]
    changed=replace(heat,bindings=tuple(replace(b,value=.04) if b.id=="alpha" else b for b in heat.bindings))
    assert heat.graph==changed.graph  # actual numeric value is in binding, not node/hash labels
    original=module(pack_graphs((heat,),vocab)).context
    different=module(pack_graphs((changed,),vocab)).context
    assert float((original-different).detach().abs().max())>1e-6
    numeric=packed.numeric.detach().requires_grad_()
    out=module(replace(packed,numeric=numeric))
    out.context[:,0].sum().backward()
    column=NUMERIC_NAMES.index("known_value")
    assert numeric.grad[1,:,column].abs().sum()>0
    assert torch.isfinite(numeric.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_graph_storage_permutation_and_operand_roles(fixture):
    specs,norm,_,_,_,vocab=fixture
    module=build_representations("C",norm,vocab).equation
    packed=pack_graphs(specs,vocab)
    order=torch.arange(14,-1,-1)
    inverse=torch.argsort(order)
    permuted=replace(packed,node_type=packed.node_type[:,order],quantity=packed.quantity[:,order],
                     status=packed.status[:,order],role=packed.role[:,order],numeric=packed.numeric[:,order],valid=packed.valid[:,order],
                     edges=inverse[packed.edges])
    before,after=module(packed),module(permuted)
    torch.testing.assert_close(before.nodes[:,order],after.nodes,rtol=1e-5,atol=2e-6)
    torch.testing.assert_close(before.context,after.context,rtol=1e-5,atol=2e-6)
    def power_spec(a,b):
        spec=specs[0];g=GraphBuilder(spec.bindings)
        u=g.make(Op.FIELD,binding="u")
        left=g.make(Op.CONSTANT,values=(a,));right=g.make(Op.CONSTANT,values=(b,))
        root=g.make(Op.EQUATION,g.make(Op.ADD,u,g.make(Op.POWER,left,right)))
        return replace(spec,graph=g.finish(root))
    left,right=power_spec(2.,3.),power_spec(3.,2.)
    assert float((module(pack_graphs((left,),vocab)).context-module(pack_graphs((right,),vocab)).context).detach().abs().max())>1e-6
    # Exact identities are encoded, rather than a per-case renumbering of fields.
    wave=pack_graphs((specs[2],),vocab)
    altered=replace(wave,quantity=torch.where(wave.quantity==vocab.index("solution.u"),vocab.index("solution.v"),wave.quantity))
    assert float((module(wave).context-module(altered).context).detach().abs().max())>1e-4


def test_observation_equivariance_pooling_padding_and_sensitivity(fixture):
    specs,norm,_,obs,_,vocab=fixture
    encoder=build_representations("C",norm,vocab).observation
    out=encoder(obs,specs)
    assert out.tokens.shape==(3,64,96) and out.summary.shape==(3,96)
    order=torch.arange(63,-1,-1)
    shuffled=encoder(permute(obs,order),specs)
    torch.testing.assert_close(shuffled.tokens,out.tokens[:,order],rtol=1e-5,atol=2e-6)
    torch.testing.assert_close(shuffled.summary,out.summary,rtol=1e-5,atol=2e-6)
    padded=encoder(pad(obs,11),specs)
    torch.testing.assert_close(padded.tokens[:,:64],out.tokens,rtol=1e-5,atol=2e-6)
    torch.testing.assert_close(padded.summary,out.summary,rtol=1e-5,atol=2e-6)
    assert (padded.tokens[:,64:]==0).all()
    values=encoder(replace(obs,value=obs.value+.3),specs)
    positions=encoder(replace(obs,position=obs.position*.9),specs)
    assert float((values.summary-out.summary).detach().abs().max())>1e-4
    assert float((positions.summary-out.summary).detach().abs().max())>1e-4


def test_absent_time_vs_zero_and_absent_storage_inert(fixture):
    specs,norm,_,obs,_,vocab=fixture
    encoder=build_representations("C",norm,vocab).observation
    original=encoder(obs,specs)
    time=obs.physical_time.clone();time[0]=float("nan")
    masked=encoder(replace(obs,physical_time=time),specs)
    torch.testing.assert_close(masked.tokens,original.tokens,rtol=0,atol=0)
    coordinate=CoordinateFeatures()
    p=torch.full((1,1,2),.3);t=torch.zeros(1,1,1);valid=torch.ones(1,1,dtype=torch.bool)
    absent=coordinate(p,t,torch.zeros_like(valid),valid)
    zero=coordinate(p,t,valid,valid)
    assert not torch.equal(absent,zero)
    assert (absent[...,30:]==0).all()  # all raw/Fourier/presence time features absent
    assert zero[0,0,-1]==1


@pytest.mark.parametrize("length",[0,64])
def test_empty_observations_memory_finite(fixture,length):
    specs,norm,_,obs,_,vocab=fixture
    module=build_representations("A",norm,vocab)
    empty=permute(obs,torch.arange(length))
    empty=replace(empty,valid=torch.zeros_like(empty.valid))
    encoded=module.observation(empty,specs)
    eq=module.equation(pack_graphs(specs,vocab))
    state=module.memory(encoded,eq.context,return_attention=True)
    assert (encoded.tokens==0).all() and (encoded.summary==0).all()
    assert torch.isfinite(state.tokens).all() and torch.isfinite(state.global_context).all()
    assert (state.attention==0).all()


def test_memory_geometry_masks_equivariance_and_retrieval_chunks(fixture):
    specs,norm,_,obs,q,vocab=fixture
    a=build_representations("A",norm,vocab)
    eq=a.equation(pack_graphs(specs,vocab));o=a.observation(obs,specs)
    state=a.memory(o,eq.context,return_attention=True)
    assert state.tokens.shape==(3,80,96) and state.attention.shape==(3,2,4,80,64)
    positions,levels=anchor_geometry()
    assert ((positions>0)&(positions<1)).all() and torch.bincount(levels).tolist()==[16,64]
    torch.testing.assert_close(state.attention.sum(-1),torch.ones(3,2,4,80),rtol=1e-6,atol=1e-6)
    assert (state.attention.masked_select((~obs.valid)[:,None,None,None,:])==0).all()
    padded=a.memory(a.observation(pad(obs,5),specs),eq.context,return_attention=True)
    torch.testing.assert_close(state.tokens,padded.tokens,rtol=1e-5,atol=4e-6)
    order=torch.arange(63,-1,-1)
    perm=a.memory(a.observation(permute(obs,order),specs),eq.context,return_attention=True)
    torch.testing.assert_close(state.tokens,perm.tokens,rtol=1e-5,atol=4e-6)
    torch.testing.assert_close(state.attention[...,order],perm.attention,rtol=1e-5,atol=2e-6)
    feature=a.query(q,specs,eq.context,o.summary)
    retrieved=a.retrieval(state,q,feature,specs)
    assert retrieved.features.shape==(3,32,96) and retrieved.indices.shape==(3,32,80)
    assert torch.equal(retrieved.indices[0,0],torch.arange(80))
    torch.testing.assert_close(retrieved.weights[:,:,:16].sum(-1),torch.ones(3,32),rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(retrieved.weights[:,:,16:].sum(-1),torch.ones(3,32),rtol=1e-6,atol=1e-6)
    parts=[];parts_q=[]
    for lo,hi in ((0,11),(11,23),(23,32)):
        chunk=type(q)(**{f.name:getattr(q,f.name)[:,lo:hi] for f in fields(q)})
        features=a.query(chunk,specs,eq.context,o.summary)
        parts_q.append(features)
        parts.append(a.retrieval(state,chunk,features,specs).features)
    torch.testing.assert_close(torch.cat(parts_q,1),feature,rtol=1e-5,atol=3e-6)
    torch.testing.assert_close(torch.cat(parts,1),retrieved.features,rtol=1e-5,atol=4e-6)


def test_gradient_paths_and_no_gradient_to_absent_time_or_padding(fixture):
    specs,norm,_,obs,q,vocab=fixture
    a=build_representations("A",norm,vocab)
    eq=a.equation(pack_graphs(specs,vocab))
    value=obs.value.detach().requires_grad_();sigma=obs.sigma.detach().requires_grad_()
    position=obs.position.detach().requires_grad_();time=obs.physical_time.detach().requires_grad_()
    o=a.observation(replace(obs,value=value,sigma=sigma,position=position,physical_time=time),specs)
    known=torch.ones(3,80,96,requires_grad=True)
    known_mask=torch.ones(3,80,dtype=torch.bool);known_mask[:,::2]=False
    state=a.memory(o,eq.context,known_context=known,known_valid=known_mask)
    qp=q.position.detach().requires_grad_();qt=q.physical_time.detach().requires_grad_()
    query=replace(q,position=qp,physical_time=qt)
    feature=a.query(query,specs,eq.context,o.summary)
    output=a.retrieval(state,query,feature,specs)
    (output.features[...,0].mean()+state.tokens[...,0].mean()+eq.context[:,1].mean()).backward()
    for tensor in (value,sigma,position,time,known,qp,qt):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum()>0
    assert (time.grad[0]==0).all() and (qt.grad[0]==0).all()
    assert (value.grad[~obs.valid]==0).all() and (sigma.grad[~obs.valid]==0).all()
    for submodule in (a.equation,a.observation,a.query,a.memory,a.retrieval):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in submodule.parameters())
        assert sum(float(p.grad.abs().sum()) for p in submodule.parameters())>0
    # Smooth query path (without discrete nearest-neighbor selection) supports second derivatives.
    q2=replace(q,position=q.position.detach().requires_grad_())
    features=a.query(q2,specs,eq.context.detach(),o.summary.detach())
    first=torch.autograd.grad(features[...,0].sum(),q2.position,create_graph=True)[0]
    second=torch.autograd.grad(first[...,0].sum(),q2.position)[0]
    assert torch.isfinite(second).all() and second.abs().sum()>0


def test_coefficient_representation_and_independent_owners(fixture):
    specs,norm,cases,_,_,vocab=fixture
    before=torch.random.get_rng_state().clone()
    modules={owner:build_representations(owner,norm,vocab) for owner in ("A","B","C")}
    assert torch.equal(before,torch.random.get_rng_state())
    ptrs=[{p.data_ptr() for p in m.parameters()} for m in modules.values()]
    assert not(ptrs[0]&ptrs[1] or ptrs[0]&ptrs[2] or ptrs[1]&ptrs[2])
    assert modules["B"].memory is None and modules["B"].retrieval is None
    assert modules["C"].memory is None
    same=build_representations("B",norm,vocab)
    assert all(torch.equal(x,y) for x,y in zip(same.parameters(),modules["B"].parameters()))
    assert not torch.equal(modules["A"].equation.numeric[0].weight,modules["B"].equation.numeric[0].weight)
    xi=case_coefficients(cases).xi.float().requires_grad_()
    gradient=torch.ones_like(xi,requires_grad=True);previous=torch.full_like(xi,.1,requires_grad=True)
    inputs=coefficient_inputs(specs,xi,gradient,previous)
    tokens,summary=modules["B"].coefficient(inputs)
    assert tokens.shape==(3,18,96) and summary.shape==(3,96)
    assert inputs.active.sum(1).tolist()==[9,9,18]
    assert (tokens[~inputs.active]==0).all()
    summary[:,0].sum().backward()
    for tensor in (xi,gradient,previous):
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum()>0
        assert (tensor.grad[~inputs.active]==0).all()
    padded=replace(inputs,xi=torch.where(inputs.active,xi,torch.full_like(xi,float("nan"))),
                   gradient=torch.where(inputs.active,gradient,torch.full_like(gradient,float("nan"))),
                   previous_update=torch.where(inputs.active,previous,torch.full_like(previous,float("nan"))))
    torch.testing.assert_close(modules["B"].coefficient(padded)[0],tokens,rtol=0,atol=0)


@pytest.mark.parametrize("device",["cpu","mps","cuda"])
def test_available_device_forward_backward_masks_and_chunks(fixture,device):
    from pdeep.runtime import capabilities,resolve_device
    if not next(c.available for c in capabilities() if c.device==device):
        pytest.skip(device+" unavailable in this process")
    resolve_device(device,"float32")
    specs,norm,cases,obs,q,vocab=fixture
    a=build_representations("A",norm,vocab).to(device)
    obs,q=float_batch(obs,device),float_batch(q,device)
    graph=a.equation(pack_graphs(specs,vocab).to(device))
    enc=a.observation(obs,specs);state=a.memory(enc,graph.context,return_attention=True)
    query=a.query(q,specs,graph.context,enc.summary)
    out=a.retrieval(state,q,query,specs)
    xi=torch.zeros(3,18,device=device,requires_grad=True)
    coeff=a.coefficient(coefficient_inputs(specs,xi,torch.ones_like(xi),torch.zeros_like(xi)))[0]
    (out.features[...,0].mean()+coeff[...,0].mean()).backward()
    assert torch.isfinite(out.features).all() and xi.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for name,module in a.named_children()
               if name!="memory" for p in module.parameters())
    with torch.no_grad():
        padded=a.memory(a.observation(pad(obs,7),specs),graph.context)
        torch.testing.assert_close(padded.tokens,state.tokens,rtol=3e-5,atol=1e-5)
        order=torch.arange(63,-1,-1,device=device)
        shuffled=a.observation(permute(obs,order),specs)
        torch.testing.assert_close(shuffled.summary,enc.summary,rtol=3e-5,atol=1e-5)
        query_pad=pad(q,5)
        features=a.query(query_pad,specs,graph.context,enc.summary)
        retrieved=a.retrieval(state,query_pad,features,specs)
        torch.testing.assert_close(retrieved.features[:,:32],out.features,rtol=3e-5,atol=1e-5)
        assert (retrieved.features[:,32:]==0).all()
        empty=permute(obs,torch.arange(0,device=device))
        empty_encoding=a.observation(empty,specs)
        empty_state=a.memory(empty_encoding,graph.context,return_attention=True)
        assert empty_state.attention.shape[-1]==0 and torch.isfinite(empty_state.tokens).all()
