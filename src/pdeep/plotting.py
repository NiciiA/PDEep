"""Reference-only plots; no plotting code samples or mutates the global RNG."""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt

from .data import (CaseDataset, SENSOR_COUNTS, case_coefficients, evaluation_grid,
                   EVALUATION_TIMES)
from .reference import canonical_spec, evaluate, evaluate_batch, prior_scales


def field_image(ax, values, title, *, limit=None):
    array = values.detach().numpy()
    limit = float(np.max(np.abs(array))) if limit is None else limit
    limit = max(limit, np.finfo(float).eps)
    image = ax.imshow(array, origin="lower", extent=(0,1,0,1), cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set(xlabel="x", ylabel="y", title=title)
    return image


def reference_fields(cases, *, resolution=64, physical_time=.2):
    fig, axes = plt.subplots(2,4,figsize=(15,7),layout="constrained")
    panels = []
    for case in cases:
        pde = case.key.pde
        query = evaluation_grid(pde,resolution=resolution,times=None if pde=="poisson" else (physical_time,))
        p = query.position[0]
        t = None if pde=="poisson" else query.physical_time[0]
        quantities = ("u","f") if pde=="poisson" else (("u","u0") if pde=="heat" else ("u","v","u0","v0"))
        for q in quantities:
            value = evaluate(pde,case.tensor(),p,None if q in {"f","u0","v0"} else t,quantity=q)
            title = f"{pde.title()}: {q}" + (f"(t={physical_time:g})" if pde!="poisson" and q in {"u","v"} else "")
            panels.append((value.reshape(resolution,resolution),title))
    for ax,(value,title) in zip(axes.flat,panels):
        image=field_image(ax,value,title)
        fig.colorbar(image,ax=ax,shrink=.8)
    fig.suptitle("Analytical T1 fields — one training case per PDE, CPU float64")
    return fig


def time_slices(case, *, quantity_id=0, resolution=64):
    """Evaluate all nine specified evaluation times, never interpolate frames."""
    query=evaluation_grid(case.key.pde,quantity_id=quantity_id,resolution=resolution)
    values=evaluate_batch((canonical_spec(case.key.pde),),case_coefficients((case,)),query)
    values=values.reshape(len(EVALUATION_TIMES),resolution,resolution)
    fig,axes=plt.subplots(3,3,figsize=(10,9),layout="constrained")
    limit=float(values.abs().max())
    for ax,time,value in zip(axes.flat,EVALUATION_TIMES,values):
        image=field_image(ax,value,f"t = {time:g}",limit=limit)
    fig.colorbar(image,ax=axes.ravel().tolist(),shrink=.65,label="v" if quantity_id else "u")
    quantity = "velocity v" if quantity_id else ("displacement u" if case.key.pde == "wave" else "field u")
    fig.suptitle(f"{case.key.pde.title()} {quantity} — direct continuous-time evaluation")
    return fig


def prior_histograms(dataset: CaseDataset, *, cases=256):
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
    groups=[]
    for pde in SENSOR_COUNTS:
        xi=np.array([dataset.case("train",pde,i).xi for i in range(cases)])
        scaled=xi/prior_scales(pde).numpy()
        groups.append((f"{pde.title()} a / tau",scaled[:,:9]))
        if pde=="wave":
            groups.append(("Wave b / (0.4 omega tau)",scaled[:,9:]))
    x=np.linspace(-4,4,300)
    for ax,(title,z) in zip(axes.flat,groups):
        ax.hist(z.flatten(),bins=35,density=True,color="#3985ac",alpha=.65,label="9 standardized modal coefficients")
        ax.plot(x,np.exp(-x*x/2)/np.sqrt(2*np.pi),color="#b64636",label="N(0,1)")
        ax.set(title=title,xlabel="standardized coefficient",ylabel="density")
        ax.legend(fontsize=8)
    fig.suptitle(f"Prior coefficient histograms — {cases} training cases per family")
    return fig


def sensor_coverage(cases, observation_sets, *, resolution=64):
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout="constrained")
    for column,(case,record) in enumerate(zip(cases,observation_sets)):
        obs=record.inputs
        pde=case.key.pde
        grid=evaluation_grid(pde,resolution=resolution,times=None if pde=="poisson" else (.2,))
        background=evaluate_batch((canonical_spec(pde),),case_coefficients((case,)),grid)
        field_image(axes[0,column],background.reshape(resolution,resolution),f"{pde.title()} — {obs.valid.sum().item()} sensors")
        xy=obs.position[0].numpy()
        if pde=="poisson":
            dots=axes[0,column].scatter(xy[:,0],xy[:,1],c=obs.value[0,:,0].numpy(),cmap="viridis",edgecolors="black",s=25)
            fig.colorbar(dots,ax=axes[0,column],shrink=.7,label="noisy u")
            axes[1,column].errorbar(record.metadata.clean.flatten().numpy(),obs.value.flatten().numpy(),
                                    yerr=obs.sigma.flatten().numpy(),fmt=".",alpha=.7,label="y ± sigma")
            lower=float(record.metadata.clean.min());upper=float(record.metadata.clean.max())
            axes[1,column].plot((lower,upper),(lower,upper),color="gray",ls="--")
            axes[1,column].set(xlabel="clean u (metadata only)",ylabel="noisy observation y",title="Stationary: no physical time")
            axes[1,column].legend()
        else:
            for q,marker in ((0,"o"),(1,"^")) if pde=="wave" else ((0,"o"),):
                mask=obs.quantity_id[0].numpy()==q
                dots=axes[0,column].scatter(xy[mask,0],xy[mask,1],c=obs.physical_time[0,mask,0].numpy(),
                                           vmin=0,vmax=1,cmap="viridis",marker=marker,edgecolors="black",s=35,label="v" if q else "u")
            fig.colorbar(dots,ax=axes[0,column],shrink=.7,label="sensor physical time")
            axes[0,column].legend(fontsize=8)
            axes[0,column].set_xlabel("x (background is u at t=0.2)")
            times=obs.physical_time.flatten().numpy()
            axes[1,column].hist(times,bins=np.linspace(0,1,21),color="#3985ac",edgecolor="white")
            axes[1,column].set(xlabel="physical time",ylabel="sensor count",title=f"zero / early / late = {record.metadata.time_counts}")
    fig.suptitle("Noisy sensors and time coverage — placement independent of hidden field values")
    return fig
