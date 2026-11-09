"""Inspectable A figures, labeled untrained/smoke-tested; never claim accuracy."""
from dataclasses import replace
import numpy as np
import torch
import matplotlib.pyplot as plt
from .a_checks import mixed_fixture,slice_tokens
from .batches import UnknownCoefficients
from .data import CaseDataset,case_coefficients,evaluation_grid,observations
from .reference import canonical_spec,evaluate_batch
from .representations import float32_batch


def architecture_diagram():
    fig,ax=plt.subplots(figsize=(12,4));ax.axis("off")
    boxes=[(.12,.76,"Typed PDE DAG + known scalars\n2 typed layers, width 96"),(.12,.24,"Noisy sensors + masks\n4 Fourier frequencies, width 96"),
           (.47,.5,"16 coarse + 64 fine anchors\n2 × 4-head sensor cross-attention\ngeometry local/global mixing"),
           (.79,.73,"All-anchor Gaussian retrieval\ncoarse ℓ=.30; fine ℓ=.15"),(.79,.25,"Pointwise 4 × 128 SiLU decoder\nindependent u/v heads × scale × boundary")]
    for x,y,text in boxes:ax.text(x,y,text,ha="center",va="center",bbox=dict(boxstyle="round,pad=.7",fc="#edf4ff",ec="#53719a"),fontsize=9)
    for a,b in [((.25,.73),(.34,.56)),((.25,.27),(.34,.44)),((.60,.55),(.66,.70)),((.79,.62),(.79,.37))]:
        ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=1.5))
    ax.text(.46,.06,"Fixed prepared context → auxiliary standardized xi head (18 masked slots)\nNo PDE solver in A; no PDE or initial-condition guarantee",ha="center",fontsize=9)
    fig.tight_layout();return fig


def memory_summary(model,normalizers,*,status="untrained"):
    device=str(next(model.parameters()).device);b=mixed_fixture(normalizers,device,queries=16)
    with torch.no_grad():context=model.prepare_context(b.specs,b.observations,return_attention=True)
    mem=context.memory;obs=context.observations
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    p=mem.positions.cpu().numpy();levels=mem.levels.cpu().numpy()
    for i,spec in enumerate(b.specs):
        ax=axes[0,i]
        norm=mem.tokens[i].norm(dim=-1).cpu().numpy()
        ax.scatter(p[levels==1,0],p[levels==1,1],c=norm[levels==1],s=35,cmap="viridis")
        ax.scatter(p[levels==0,0],p[levels==0,1],marker="s",s=80,facecolors="none",edgecolors="black",label="coarse")
        valid=obs.valid[i].cpu();positions=obs.position[i,valid].cpu().numpy()
        ax.scatter(positions[:,0],positions[:,1],marker="x",c="red",label="sensors")
        ax.set(title=f"{spec.name}: anchor norms",xlim=(0,1),ylim=(0,1),aspect="equal");ax.legend(fontsize=7)
        selected=[0,5,16,43,79]
        weights=mem.attention[i,-1,:,selected].mean(0)[:,obs.valid[i]].cpu().numpy()
        axes[1,i].imshow(weights,aspect="auto",cmap="magma",interpolation="nearest")
        axes[1,i].set(title="Final-layer head mean; selected anchors",xlabel="Valid sensor",ylabel="Anchor",yticks=range(5),yticklabels=selected)
    fig.suptitle(f"Architecture A — {status}; attention is not physical influence")
    return fig


def field_examples(model,normalizers,*,status="untrained",resolution=64):
    dataset=CaseDataset(normalizers.root_seed);device=str(next(model.parameters()).device)
    fig,axes=plt.subplots(4,4,figsize=(13,12),constrained_layout=True)
    entries=(("poisson",0),("heat",0),("wave",0),("wave",1))
    for row,(name,quantity) in enumerate(entries):
        case=dataset.case("train",name,7);spec=canonical_spec(name)
        obs=observations(dataset,case,normalizers,set_id="A-plot").inputs
        query=evaluation_grid(name,quantity_id=quantity,resolution=resolution,times=None if name=="poisson" else (.2,))
        truth=evaluate_batch((spec,),case_coefficients((case,)),query)[0,:,0].reshape(resolution,resolution).numpy()
        with torch.no_grad():
            context=model.prepare_context((spec,),float32_batch(obs,device=device))
            fq=float32_batch(query,device=device)
            pred=torch.cat([model.predict_queries(context,slice_tokens(fq,j,min(j+512,len(truth.ravel())))).predicted_value for j in range(0,len(truth.ravel()),512)],1)
        pred=pred[0,:,0].cpu().numpy().reshape(resolution,resolution);limit=max(abs(truth).max(),abs(pred).max(),1e-12)
        for col,(data,title) in enumerate(((truth,"Ground truth"),(pred,"Learned field"),(pred-truth,"Prediction − truth"))):
            im=axes[row,col].imshow(data,origin="lower",extent=(0,1,0,1),cmap="RdBu_r",vmin=-limit,vmax=limit)
            axes[row,col].set_title(f"{name} {'u' if quantity==0 else 'v'} — {title}",fontsize=9)
            fig.colorbar(im,ax=axes[row,col],shrink=.7)
        mask=obs.quantity_id[0]==quantity;p=obs.position[0,mask].numpy();values=obs.value[0,mask,0].numpy()
        ax=axes[row,3];sc=ax.scatter(p[:,0],p[:,1],c=values,cmap="RdBu_r",vmin=-limit,vmax=limit,s=35)
        ax.set(title="Sparse noisy sensors (all times)",xlim=(0,1),ylim=(0,1),aspect="equal")
        fig.colorbar(sc,ax=ax,shrink=.7)
        if name!="poisson":ax.set_xlabel("Fields at t=.2; sensor times differ",fontsize=8)
    fig.suptitle(f"Architecture A — {status}. Execution diagnostic; reconstruction accuracy not established.")
    return fig


def training_curve(report):
    logs=report["updates"];x=[l["update"] for l in logs]
    fig,axes=plt.subplots(1,2,figsize=(10,3),constrained_layout=True)
    for key in ("field","xi","total","rolling_total_loss"):axes[0].plot(x,[l[key] for l in logs],"o-",label=key)
    axes[0].legend();axes[0].set(xlabel="Optimizer update s",ylabel="Objective",title=report["status"])
    axes[1].plot(x,[l["gradient_norm_before_clip"] for l in logs],"o-",label="before clipping")
    axes[1].plot(x,[l["gradient_norm_after_clip"] for l in logs],"o-",label="after clipping")
    axes[1].legend();axes[1].set(xlabel="Optimizer update s",ylabel="Gradient L2 norm")
    return fig


def fixed_query_comparison(model,batch,*,status):
    """Opt-in overfit diagnostics on its actual fixed supervised query set."""
    with torch.no_grad():prediction=model(batch.specs,batch.observations,batch.queries).predicted_value
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for q,label in ((0,"u"),(1,"v")):
        mask=batch.queries.valid&(batch.queries.quantity_id==q)
        truth=batch.target[mask].cpu().numpy().ravel();pred=prediction[mask].cpu().numpy().ravel()
        axes[q].scatter(truth,pred,s=5,alpha=.5)
        if len(truth):
            lo,hi=min(truth.min(),pred.min()),max(truth.max(),pred.max());axes[q].plot([lo,hi],[lo,hi],"k--")
        axes[q].set(title=f"Fixed training queries: {label}",xlabel="Ground truth (physical units)",ylabel="A prediction")
    fig.suptitle(status);return fig
