"""Read-only coordinate-layout/stripe audit, with no plot interpolation in metrics."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from ..core import require
from ..artifact_io import atomic_json
from ..artifact_tensors import state_identity
from ..geometric.data import Tokens
from ..smooth.definition import random_stream
from .reference_cache import FixedReference
from .metrics import moments


def grid_layout(mesh,size):
    lo,hi=mesh.vertices.min(0),mesh.vertices.max(0)
    xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],size),np.linspace(lo[1],hi[1],size),indexing='ij')
    x=np.column_stack((xx.ravel(),yy.ravel()));features=mesh.features(x)
    mask=features['inside']&(features['distance']>1e-7)
    return x[mask],mask.reshape(size,size),lo,hi


def directional_grid(x,pred,truth,scale,span=None):
    ux,uy=np.unique(x[:,0]),np.unique(x[:,1]);P=np.full((len(ux),len(uy)),np.nan);Y=P.copy()
    i=np.searchsorted(ux,x[:,0]);j=np.searchsorted(uy,x[:,1]);P[i,j]=pred/scale;Y[i,j]=truth/scale;E=P-Y
    result={}
    for k,axis in ((0,'x'),(1,'y')):
        delta=np.diff(E,axis=k);result[axis+'_neighbor_difference_energy']=float(np.nanmean(delta**2))
        h=(ux[1]-ux[0]) if k==0 else (uy[1]-uy[0])
        result[axis+'_error_gradient_energy']=float(np.nanmean((delta/h)**2))
        second=np.diff(E,n=2,axis=k);result[axis+'_second_difference_energy']=float(np.nanmean(second**2));result[axis+'_error_curvature_energy']=float(np.nanmean((second/h**2)**2))
    if np.isfinite(E).all() and min(E.shape)>=8:
        window=np.outer(np.hanning(len(ux)),np.hanning(len(uy)));power=np.abs(np.fft.fft2((E-E.mean())*window))**2
        # Frequencies in cycles per physical bounding-box extent, matching query
        # Fourier features. Do not apply FFT to zero-filled holes/irregular masks.
        span=np.asarray(span) if span is not None else np.array([ux[-1]-ux[0],uy[-1]-uy[0]])
        fx=np.fft.fftfreq(len(ux),d=(ux[1]-ux[0])/span[0]);fy=np.fft.fftfreq(len(uy),d=(uy[1]-uy[0])/span[1])
        X,Z=np.meshgrid(fx,fy,indexing='ij');total=power.sum()
        hf=(abs(X)>=8)|(abs(Z)>=8);stripe=((abs(X)>=8)&(abs(Z)<=1))|((abs(Z)>=8)&(abs(X)<=1))
        order=np.argsort(power.ravel())[-10:][::-1]
        normalization=E.size*float((window**2).sum())
        hf_order=np.argsort(np.where(hf,power,0).ravel())[-8:][::-1]
        result['spectrum']={'windowed_error_MSE':float(total/normalization),'HF_error_MSE':float(power[hf].sum()/normalization),'axis_HF_error_MSE':float(power[stripe].sum()/normalization),'largest_HF_bins':[{'x_cycles':float(X.ravel()[a]),'y_cycles':float(Z.ravel()[a]),'fraction':float(power.ravel()[a]/total) if total else 0.} for a in hf_order],'HF_error_energy_fraction':float(power[hf].sum()/total) if total else 0.,'axis_aligned_HF_error_fraction':float(power[stripe].sum()/total) if total else 0.,'largest_error_bins':[{'x_cycles':float(X.ravel()[a]),'y_cycles':float(Z.ravel()[a]),'fraction':float(power.ravel()[a]/total) if total else 0.} for a in order],'method':'mean-removed Hann window; full interior rectangle only, no hole zero-filling'}
    else:result['spectrum']={'status':'not computed: irregular/masked domain; use valid-neighbor and random interior derivatives'}
    return result


def audit_striping(model,data,directory,*,reference_cache=None,grid_sizes=(64,128),random_count=4096,native_count=4096,quantity='u',physical_time=.25):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);before=state_identity(model);previous=model.training;model.eval()
    case=data.case;mesh=case.problem.mesh;norm=data.normalizers.scale(case.pde,quantity);q=1 if quantity=='v' else 0
    require(quantity=='u' or case.pde=='wave' and quantity=='v','invalid stripe-audit quantity')
    ref=reference_cache or FixedReference(case);device=next(model.parameters()).device
    tvalue=0. if case.pde=='poisson' else physical_time
    obs=data.observations.to(device)
    def queries(x):
        x=torch.as_tensor(x,device=device,dtype=torch.float32)[None];n=x.shape[1]
        return Tokens(x,torch.full((1,n,1),tvalue,device=device),torch.full((1,n),case.pde!='poisson',device=device,dtype=torch.bool),torch.full((1,n),q,device=device,dtype=torch.long),torch.ones((1,n),device=device,dtype=torch.bool))
    try:
        with torch.no_grad():
            contexts=model.prepare_context((case.problem,),obs);context=contexts[0]
            def predict(x,chunk=512,order=None):
                tokens=queries(x)
                if order is not None:
                    tokens=Tokens(**{k:None if v is None else v[:,order] for k,v in vars(tokens).items()})
                values=model.predict(contexts,tokens,chunk=chunk)[0,:,0].cpu().numpy()
                return values if order is None else values[np.argsort(order)]
            layouts={};raw={}
            for size in grid_sizes:
                x,mask,lo,hi=grid_layout(mesh,size);raw['grid_'+str(size)]=x
            raw['random']=mesh.sample(random_count,random_stream(case.case_id,'stripe-audit-random'))
            native=ref.reference.space.position;native=native[mesh.features(native)['distance']>1e-7]
            ids=random_stream(case.case_id,'stripe-native-nodes').choice(len(native),min(native_count,len(native)),replace=False);raw['reference_mesh']=native[ids]
            for name,x in raw.items():
                y=ref.evaluate(x,np.full(len(x),tvalue),np.full(len(x),q));p=predict(x)
                row=moments(torch.tensor(p),torch.tensor(y),norm)
                if name.startswith('grid'):row['directional_error']=directional_grid(x,p,y,norm,np.ptp(mesh.vertices,axis=0))
                layouts[name]={'points':len(x),'metrics':row,'measure':'equal grid points' if name.startswith('grid') else 'uniform physical area' if name=='random' else 'sampled P4 reference DOFs, NOT area-uniform'}
                np.savez_compressed(directory/(name+'.npz'),position=x,truth=y,prediction=p)
            x=raw['random'][:min(256,random_count)];base=predict(x,256)
            changes={}
            for chunk in (1,31,256):changes['chunk_'+str(chunk)]=float(np.max(abs(predict(x,chunk)-base))/norm)
            perm=random_stream(case.case_id,'stripe-order').permutation(len(x));changes['query_order']=float(np.max(abs(predict(x,31,perm)-base))/norm)
            frequencies=model.coordinates.fourier.frequencies.cpu().numpy().copy()
            # Read-only feature-path interventions; never edit weights or defaults.
            x=raw['grid_'+str(max(grid_sizes))];y=ref.evaluate(x,np.full(len(x),tvalue),np.full(len(x),q))
            def suppress(module,inputs,output):
                n=inputs[0].shape[-1];mask=torch.cat((module.frequencies<8,module.frequencies<8)).repeat(n).to(output.dtype)
                return output*mask
            handle=model.coordinates.fourier.register_forward_hook(suppress)
            try:low=predict(x)
            finally:handle.remove()
            memory_zero=replace(context,memory=torch.zeros_like(context.memory))
            zeros=model.predict((memory_zero,),queries(x),chunk=512)[0,:,0].cpu().numpy()
            interventions={label:{'metrics':moments(torch.tensor(p),torch.tensor(y),norm),'directional_error':directional_grid(x,p,y,norm,np.ptp(mesh.vertices,axis=0))} for label,p in [('query_Fourier_ge8_suppressed',low),('local_memory_zeroed',zeros)]}
            # Geometry-valid random finite differences: reject pairs near boundaries.
            xr=raw['random'];h=.001*np.sqrt(mesh.cell_areas.sum());keep=mesh.features(xr)['distance']>2*h;xr=xr[keep][:256]
            gradients={}
            for axis in (0,1):
                offset=np.zeros_like(xr);offset[:,axis]=h
                pp,pm=predict(xr+offset),predict(xr-offset)
                yp=ref.evaluate(xr+offset,np.full(len(xr),tvalue),np.full(len(xr),q));ym=ref.evaluate(xr-offset,np.full(len(xr),tvalue),np.full(len(xr),q))
                gradients['xy'[axis]]={'error_gradient_energy':float(np.mean(((pp-pm-yp+ym)/(2*h*norm))**2)),'prediction_gradient_energy':float(np.mean(((pp-pm)/(2*h*norm))**2)),'reference_gradient_energy':float(np.mean(((yp-ym)/(2*h*norm))**2))}
            from scipy.spatial import cKDTree
            anchor_xy=np.unique(mesh.vertices[context.layout.anchors],axis=0)
            spacing=float(np.median(cKDTree(anchor_xy).query(anchor_xy,k=2)[0][:,1]))
            result={'anchor_median_nearest_spacing':spacing,'model_state_identity':before,'case_id':case.case_id,'observation_identity':data.identities['observation'],'reference_identity':ref.identity,'quantity':quantity,'physical_time':None if case.pde=='poisson' else tvalue,'layouts':layouts,'invariance_max_standardized_errors':changes,'Fourier_cycles_per_normalized_axis':frequencies.tolist(),'anchor_counts':context.layout.counts,'anchors':mesh.vertices[context.layout.anchors].tolist(),'random_directional_differences':gradients,'read_only_interventions':interventions,'interpretation':'Path interventions are out-of-distribution diagnostics, NOT architecture fixes or evidence that resampling has fixed striping. Quantitative metrics use raw point predictions; plotting performs no interpolation.'}
            require(max(changes.values())<2e-5,'query chunk/order discrepancy exceeds float32 control tolerance')
            atomic_json(directory/'striping.json',result)
            plot_layouts(directory,raw,norm,case,quantity)
            return result
    finally:model.train(previous);require(state_identity(model)==before,'striping audit changed parameters')


def plot_layouts(directory,layouts,scale,case,quantity):
    import matplotlib.pyplot as plt
    names=list(layouts);fig,axs=plt.subplots(len(names),3,figsize=(11,3*len(names)),squeeze=False,layout='constrained')
    packs=[np.load(directory/(n+'.npz')) for n in names];lo=min(min(p['truth'].min(),p['prediction'].min()) for p in packs);hi=max(max(p['truth'].max(),p['prediction'].max()) for p in packs)
    for row,(name,p) in enumerate(zip(names,packs)):
        for col,(label,values) in enumerate([('truth',p['truth']),('prediction',p['prediction']),('absolute error',abs(p['truth']-p['prediction']))]):
            sc=axs[row,col].scatter(*p['position'].T,c=values,s=3,cmap='viridis' if col<2 else 'magma',vmin=lo if col<2 else 0,vmax=hi if col<2 else None)
            axs[row,col].set(aspect='equal',title=f'{name} · {quantity} · {label}');fig.colorbar(sc,ax=axs[row,col],shrink=.7)
    fig.savefig(directory/'coordinate-layouts.png',dpi=140);plt.close(fig)
