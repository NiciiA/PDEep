"""Quantity-first reconstruction, collapse, gradient and conditioning diagnostics."""
import math
import numpy as np
import torch
from ..geometric.representation import retrieval_weights, tensor
from ..geometric.physics import supervised, weak_loss


def scalar(x): return float(x.detach())
def moments(pred,truth,scale):
    pred,truth=pred.flatten(),truth.flatten(); e=pred-truth
    pr=pred.square().mean().sqrt(); tr=truth.square().mean().sqrt()
    return {'standardized_MSE':scalar((e/scale).square().mean()),'standardized_RMSE':scalar((e/scale).square().mean().sqrt()),
            'physical_RMSE':scalar(e.square().mean().sqrt()),'amplitude_ratio':scalar(pr/tr) if tr>0 else None,
            'prediction_RMS':scalar(pr),'truth_RMS':scalar(tr),'prediction_mean':scalar(pred.mean()),'truth_mean':scalar(truth.mean()),
            'mean_bias':scalar(e.mean()),'prediction_variance':scalar(pred.var(unbiased=False)),'truth_variance':scalar(truth.var(unbiased=False))}


def quantity_metrics(pred,query,pde,norms):
    out={}
    for q in ((0,1) if pde=='wave' else (0,)):
        valid=query.valid[0] & (query.quantity_id[0]==q); name='v' if q else 'u'
        if not valid.any(): continue
        s=norms.scale(pde,q); t=query.physical_time[0,:,0]
        row=moments(pred[0,valid],query.value[0,valid],s)
        if pde!='poisson':
            masks={'t0':t==0,'early':(t>0)&(t<=.1),'middle':(t>.1)&(t<=.5),'late':(t>.5)&(t<=1)}
            row['time_groups']={k:moments(pred[0,valid&m],query.value[0,valid&m],s) for k,m in masks.items() if (valid&m).any()}
            row['per_time']={str(float(v)):moments(pred[0,valid&(t==v)],query.value[0,valid&(t==v)],s) for v in t[valid].unique()}
            # Grid blocks share physical locations at every time: variation across t,
            # separate from variation across x at a fixed t.
            times=t[valid].unique(); slices=[pred[0,valid&(t==v),0] for v in times]
            if len({len(v) for v in slices})==1:
                pp=torch.stack(slices); yy=torch.stack([query.value[0,valid&(t==v),0] for v in times])
                row['temporal_variance_prediction']=scalar(pp.var(0,unbiased=False).mean()); row['temporal_variance_truth']=scalar(yy.var(0,unbiased=False).mean())
                row['spatial_variance_prediction']=scalar(pp.var(1,unbiased=False).mean()); row['spatial_variance_truth']=scalar(yy.var(1,unbiased=False).mean())
        # Grid-neighbor error differences expose spatially oscillatory errors;
        # this diagnostic is NOT an additional training term or PDE residual.
        from scipy.spatial import cKDTree
        spatial=[]
        for time_value in t[valid].unique() if pde!='poisson' else [0.]:
            mask=valid & (t==time_value) if pde!='poisson' else valid
            xx=query.position[0,mask].detach().cpu().numpy()
            if len(xx)<3:continue
            distance,near=cKDTree(xx).query(xx,k=2)
            # Nearest-grid links; includes no boundary/ghost samples.
            jj=torch.as_tensor(near[:,1],device=pred.device);hh=torch.as_tensor(distance[:,1],device=pred.device,dtype=pred.dtype)
            err=((pred[0,mask,0]-query.value[0,mask,0])/s)
            spatial.append(scalar(((err-err[jj])/hh).square().mean()))
        row['spatial_error_neighbor_gradient_energy']=float(np.mean(spatial)) if spatial else None
        row['high_frequency_diagnostic_scope']='fixed-grid nearest-neighbor standardized error differences per physical length; proxy, not a spectral/PDE norm'
        out[name]=row
    return out


def reconstruction(model,data,*,chunk=512):
    previous=model.training; model.eval(); b=data.batch(next(model.parameters()).device); d=data.batch(b.observations.position.device,diagnostic=True)
    try:
        with torch.no_grad():
            cs=model.prepare_context(b.specs,b.observations)
            train=model.predict(cs,b.queries,chunk=chunk); pred=model.predict(cs,d.queries,chunk=chunk); sensors=model.predict(cs,b.observations,chunk=chunk)
            result={'training':quantity_metrics(train,b.queries,data.case.pde,data.normalizers), 'field':quantity_metrics(pred,d.queries,data.case.pde,data.normalizers),
                    'sensors':quantity_metrics(sensors,b.observations,data.case.pde,data.normalizers),'query_identity':data.identities['diagnostic'],'fixed_data_identity':data.identity}
            from ..geometric.data import Tokens
            clean=Tokens(**{**vars(b.observations),'value':data.clean_observations.to(sensors.device)})
            result['clean_sensors']=quantity_metrics(sensors,clean,data.case.pde,data.normalizers)
            x=b.queries.position[0,b.queries.valid[0]]; t=x.new_zeros((len(x),1)) if data.case.pde!='poisson' else None
            result['auxiliary']={name:moments(model.evaluate(cs[0],x,t,name),tensor(field(x.cpu().double().numpy()),x),data.normalizers.scale(data.case.pde,name)) for name,field in data.case.fields.items()}
            from ..geometric.quadrature import boundary
            qb=boundary(data.case.problem.mesh,order=3); bx=tensor(qb.position,x); w=tensor(qb.weight,x); br={}
            for name in ('u','v') if data.case.pde=='wave' else ('u',):
                vals=[]
                for time in (0.,.05,.25,.5,1.) if t is not None else (0.,):
                    v=model.evaluate(cs[0],bx,None if t is None else bx.new_full((len(bx),1),time),name)[:,0]
                    vals.append(scalar((w*(v/data.normalizers.scale(data.case.pde,name)).square()).sum()/w.sum()))
                br[name]={'standardized_boundary_MSE':float(np.mean(vals)),'times':vals,'policy':'tagged homogeneous Dirichlet; physical boundary measure'}
            result['boundary']=br
        return result
    finally:model.train(previous)


def gradient_terms(model,batch,adapter):
    """Independent per-loss autograd, NO backward accumulation or optimizer step."""
    report=adapter.loss(model,batch,detailed=True); named=list(model.named_parameters()); params=[p for _,p in named]
    vectors={}; modules={}; grads={}
    for name,loss in report.weighted.items():
        g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        vec=torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten().cpu() for p,v in zip(params,g)])
        vectors[name]=vec; grads[name]=float(vec.norm()); modules[name]={}
        for (pn,p),v in zip(named,g):
            top=pn.split('.')[0]; modules[name][top]=modules[name].get(top,0.)+(0 if v is None else scalar(v.square().sum()))
        modules[name]={k:math.sqrt(v) for k,v in modules[name].items()}
    cosine={f'{a}:{b}':float(vectors[a]@vectors[b]/(vectors[a].norm()*vectors[b].norm())) if vectors[a].norm()*vectors[b].norm()>0 else None for a in vectors for b in vectors if a<b}
    return {'raw':{k:scalar(v) for k,v in report.unweighted.items()},'weighted':{k:scalar(v) for k,v in report.weighted.items()},'total':scalar(report.total),'weighted_gradient_norms':grads,'per_module_weighted_gradient_norms':modules,'weighted_gradient_cosines':cosine,'optimizer_updates':0}


def representation(model,data):
    b=data.batch(next(model.parameters()).device); captured=[]; handles=[]
    if model.config.architecture=='C_GEO':
        for layer in model.decoder.layers:
            handles.append(layer.modulation.register_forward_hook(lambda m,i,o:captured.append(o.detach())))
    try:
        cs=model.prepare_context(b.specs,b.observations); c=cs[0]
        x=b.queries.position[0,:min(12,b.queries.position.shape[1])].clone().requires_grad_(True)
        t=b.queries.physical_time[0,:len(x)].clone().requires_grad_(True) if data.case.pde!='poisson' else None
        pred=model.evaluate(c,x,t,'u')/data.normalizers.scale(data.case.pde,'u')
        dx=torch.autograd.grad(pred.sum(),x,retain_graph=True)[0]
        w=retrieval_weights(c.layout,x); nc=len(c.layout.coarse)
        r={'coordinates':'physical xy + extent-normalized Fourier xy; physical time; geometry-owned boundary distance/normals',
           'extent':np.ptp(c.problem.mesh.vertices,axis=0).tolist(),'scales':data.normalizers.values[data.case.pde],
           'anchors':c.layout.counts,'query_coordinate_gradient_RMS':scalar(dx.square().mean().sqrt()),
           'memory_token_variance':scalar(c.memory.var(0,unbiased=False).mean()),'geometry_token_variance':scalar(c.geometry_nodes.var(0,unbiased=False).mean()),
           'global_context_RMS':scalar(c.global_context.square().mean().sqrt()),'latent_RMS':scalar(c.latent.square().mean().sqrt()),
           'local_coarse_RMS':scalar((w[:,:nc]@c.memory[:nc]).square().mean().sqrt()),'local_fine_RMS':scalar((w[:,nc:]@c.memory[nc:]).square().mean().sqrt()),
           'retrieval_scale_sum_errors':[scalar((w[:,:nc].sum(-1)-1).abs().max()),scalar((w[:,nc:].sum(-1)-1).abs().max())],
           'attention':[{'shape':list(a.shape),'entropy':scalar(-(a.clamp_min(1e-30).log()*a).sum(-1).mean()),'variance':scalar(a.var(unbiased=False))} for a in c.attention if a.numel()],
           'retrieval_entropy':scalar(-(w.clamp_min(1e-30).log()*w).sum(-1).mean()),'FiLM':[]}
        for z in captured:
            r['FiLM'].append({'modulation_RMS':scalar((.1*z.tanh()).square().mean().sqrt()),'tanh_saturation_fraction':scalar((z.tanh().abs()>.99).float().mean()),'conditioning_variance':scalar(z.var(unbiased=False)),'conditioning_spatial_variance':scalar(z.var(0,unbiased=False).mean())})
        from dataclasses import replace
        with torch.no_grad():
            r['context_path_ablation']={}
            for label,altered in [('zero_local_memory',replace(c,memory=torch.zeros_like(c.memory))),('zero_direct_global',replace(c,global_context=torch.zeros_like(c.global_context))),('zero_latent',replace(c,latent=torch.zeros_like(c.latent)))]:
                other=model.evaluate(altered,x,t,'u')/data.normalizers.scale(data.case.pde,'u')
                r['context_path_ablation'][label]={'standardized_prediction_RMS_change':scalar((other-pred).square().mean().sqrt()),'scope':'diagnostic path ablation at fixed queries; latent/global/local remaining paths retained, not a valid new physical conditioning set'}
        # Module condition/local contribution cannot be inferred from gradient alone;
        # report feature variance and observation perturbation response separately.
        if data.case.pde=='wave': r['wave_consistency']=wave_consistency(model,c)
        return r
    finally:
        for h in handles:h.remove()


def wave_consistency(model,c):
    # Centroids with >0 barycentric margin stay in one element under FD steps.
    mesh=c.problem.mesh; ids=np.linspace(0,len(mesh.cells)-1,8,dtype=int)
    x=tensor(mesh.vertices[mesh.cells[ids]].mean(1),c.equation).requires_grad_(True)
    t=x.new_full((len(x),1),.25,requires_grad=True)
    u=model.evaluate(c,x,t,'u'); v=model.evaluate(c,x,t,'v')
    ut=torch.autograd.grad(u.sum(),t,create_graph=True,retain_graph=True)[0]
    vt=torch.autograd.grad(v.sum(),t,create_graph=True,retain_graph=True)[0]
    dx=torch.autograd.grad(u.sum(),x,create_graph=True,retain_graph=True)[0]
    lap=sum(torch.autograd.grad(dx[:,k].sum(),x,create_graph=True,retain_graph=True)[0][:,k:k+1] for k in range(2))
    eps=1e-4
    with torch.no_grad():fd=(model.evaluate(c,x,t+eps,'u')-model.evaluate(c,x,t-eps,'u'))/(2*eps)
    with torch.no_grad():
        h=2e-4;spatial_fd=[]
        for k in range(2):
            offset=torch.zeros_like(x);offset[:,k]=h
            # Assert the finite differences remain in the same interpolation cell.
            center_ids=mesh.locate(x.detach().cpu().double().numpy())[0]
            assert np.array_equal(center_ids,mesh.locate((x+offset).cpu().double().numpy())[0])
            assert np.array_equal(center_ids,mesh.locate((x-offset).cpu().double().numpy())[0])
            spatial_fd.append((model.evaluate(c,x+offset,t,'u')-model.evaluate(c,x-offset,t,'u'))/(2*h))
        spatial_fd=torch.cat(spatial_fd,-1)
    return {'spatial_gradient_AD_FD_max_error':scalar((dx-spatial_fd).abs().max()),'spatial_gradient_AD_max':scalar(dx.abs().max()),'spatial_FD_step':h,'R1_physical_RMS':scalar((ut-v).square().mean().sqrt()),'R2_element_interior_physical_RMS':scalar((vt-c.problem.coefficient*lap).square().mean().sqrt()),'ut_AD_FD_max_error':scalar((ut-fd).abs().max()),'FD_dt':eps,'scope':'8 triangle centroids at physical t=.25; R2 element-interior diagnostic ONLY, not a globally C2 strong PDE residual; supervised errors separate'}


def physical_diagnostics(model,data,config):
    """Same weak/BC/initial measurements for BOTH C objectives, diagnostic only."""
    previous=model.training;model.eval()
    try:
        b=data.batch(next(model.parameters()).device)
        with torch.enable_grad():
            contexts=model.prepare_context(b.specs,b.observations)
            r=weak_loss(model,contexts,data.normalizers,config)
            return {'unweighted':{k:scalar(v) for k,v in r.items() if k!='diagnostics'},'scope':'diagnostic only; not added to training objective; first weak derivatives; predicted auxiliary fields','details':r['diagnostics']}
    finally:model.train(previous)
