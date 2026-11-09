"""D-specific figures. All interval labels distinguish latent from noisy values."""
import numpy as np
import torch
import matplotlib.pyplot as plt

from .data import evaluation_grid
from .reference import evaluate, prior_scales
from .plotting import field_image
from .batches import QueryBatch


def coefficient_posteriors(cases,posteriors):
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
    panels=[]
    for case,posterior in zip(cases,posteriors):
        panels.append((case,posterior,slice(0,9),"a"))
        if case.key.pde=="wave":panels.append((case,posterior,slice(9,18),"b"))
    x=np.arange(9)
    for ax,(case,posterior,section,block) in zip(axes.flat,panels):
        scales=prior_scales(case.key.pde)[section]
        mean=(posterior.mean[section]/scales).numpy()
        std=(posterior.covariance.diag()[section].sqrt()/scales).numpy()
        truth=(case.tensor()[section]/scales).numpy()
        ax.axhspan(-2,2,color="gray",alpha=.15,label="prior mean ± 2 prior SD")
        ax.errorbar(x,mean,yerr=2*std,fmt="o",color="#2378a0",capsize=3,label="analytic posterior mean ± 2 SD")
        ax.scatter(x,truth,marker="x",s=55,color="#b13a2f",label="true coefficients")
        ax.set(title=f"{case.key.pde.title()} — {block}",xlabel="lexicographic mode index",ylabel="coefficient / prior SD")
        ax.legend(fontsize=8)
    fig.suptitle("D exact coefficient posterior — physical outputs standardized only for this plot")
    return fig


def reconstructed_fields(cases,posteriors,conditioning,*,resolution=64,physical_time=.2):
    rows=[]
    for case,posterior,obs in zip(cases,posteriors,conditioning):
        for quantity in (0,1) if case.key.pde=="wave" else (0,):
            grid=evaluation_grid(case.key.pde,quantity_id=quantity,resolution=resolution,
                                 times=None if case.key.pde=="poisson" else (physical_time,))
            pred=posterior.predict(grid)
            true=evaluate(case.key.pde,case.tensor(),grid.position[0],
                          None if case.key.pde=="poisson" else grid.physical_time[0],quantity="v" if quantity else "u")
            rows.append((case,obs,quantity,true.reshape(resolution,resolution),pred))
    fig,axes=plt.subplots(len(rows),4,figsize=(17,3.5*len(rows)),layout="constrained")
    for row,(case,obs,quantity,true,pred) in enumerate(rows):
        mean=pred.mean.reshape(resolution,resolution)
        std=pred.latent_variance.sqrt().reshape(resolution,resolution)
        limit=float(torch.stack((true.abs().max(),mean.abs().max())).max())
        label=f"{case.key.pde.title()} {'v' if quantity else 'u'}"
        if case.key.pde!="poisson":label+=f", t={physical_time:g}"
        for column,(field,title) in enumerate(((true,label+": truth"),(mean,"Posterior mean"),((mean-true).abs(),"Absolute mean error"),(std,"Latent posterior SD"))):
            if column<2:
                image=field_image(axes[row,column],field,title,limit=limit)
            else:
                image=axes[row,column].imshow(field.numpy(),origin="lower",extent=(0,1,0,1),cmap="viridis",vmin=0.)
                axes[row,column].set(title=title,xlabel="x",ylabel="y")
            fig.colorbar(image,ax=axes[row,column],shrink=.75)
        mask=obs.quantity_id[0]==quantity
        xy=obs.position[0,mask].numpy()
        axes[row,0].scatter(xy[:,0],xy[:,1],s=12,facecolors="none",edgecolors="black",
                            label="sensor locations" if case.key.pde=="poisson" else "sensor locations (all t)")
        axes[row,0].legend(fontsize=7)
    fig.suptitle("Exact D reconstruction and pointwise latent uncertainty — validation case 0 per PDE")
    return fig


def heldout_predictions(cases,posteriors,conditioning,heldout,normalizers):
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout="constrained")
    for col,(case,posterior,obs,hold) in enumerate(zip(cases,posteriors,conditioning,heldout)):
        pde=case.key.pde
        for row,record,title in ((0,obs,"Conditioning set: replicate diagnostic"),(1,hold,"Independent held-out sensor set")):
            pred=posterior.predict(record.inputs,sigma=record.inputs.sigma)
            mean=pred.mean[0,:,0].numpy()
            sd=pred.variance[0,:,0].sqrt().numpy()
            target=record.inputs.value[0,:,0].numpy()
            # Display normalization uses the existing training-only calibrator.
            scales=np.ones_like(target)
            if pde=="wave":
                q=record.inputs.quantity_id[0].numpy()
                for quantity in (0,1):
                    scales[q==quantity]=normalizers.scale(pde,quantity)
            axes[row,col].errorbar(target/scales,mean/scales,yerr=2*sd/scales,fmt=".",capsize=2,alpha=.75)
            lo=float(min((target/scales).min(),(mean/scales).min()));hi=float(max((target/scales).max(),(mean/scales).max()))
            axes[row,col].plot((lo,hi),(lo,hi),ls="--",color="gray")
            suffix=" / train RMS" if pde=="wave" else ""
            axes[row,col].set(title=f"{pde.title()} — {title}",xlabel="noisy measured y"+suffix,
                              ylabel="posterior mean ± 2 predictive SD"+suffix)
    fig.suptitle("Noisy measurement predictions — predictive intervals include declared measurement variance")
    return fig


def identifiability(posteriors):
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout="constrained")
    for ax,posterior in zip(axes,posteriors):
        values=posterior.diagnostics["noise_and_prior_whitened_H"]["singular_values"]
        ax.semilogy(np.arange(1,len(values)+1),np.maximum(values,np.finfo(float).tiny),"o-")
        ax.axhline(1.,color="gray",ls="--",label="data information = prior precision")
        ax.set(title=f"{posterior.spec.name.title()} — rank {posterior.diagnostics['H']['rank']}/{len(values)}",
               xlabel="singular value index",ylabel="singular value of R^(-1/2) H L0")
        ax.legend(fontsize=7)
    fig.suptitle("Identifiability diagnostics in prior-standardized coefficient coordinates")
    return fig


def coherent_sample_traces(cases,posteriors,banks):
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
    panels=[]
    for case,posterior,bank in zip(cases,posteriors,banks):
        for quantity in (0,1) if case.key.pde=="wave" else (0,):
            panels.append((case,posterior,bank,quantity))
    axis=torch.linspace(0.,1.,201,dtype=torch.float64)
    for ax,(case,posterior,bank,quantity) in zip(axes.flat,panels):
        pde=case.key.pde
        position=torch.full((1,len(axis),2),.61,dtype=torch.float64)
        time=axis[None,:,None] if pde!="poisson" else torch.zeros(1,len(axis),1,dtype=torch.float64)
        if pde=="poisson":position[0,:,0]=axis
        else:position[0,:,0]=.37
        query=QueryBatch(position,time,torch.full((1,len(axis)),pde!="poisson",dtype=torch.bool),
                         torch.full((1,len(axis)),quantity,dtype=torch.int64),torch.ones(1,len(axis),dtype=torch.bool))
        pred=posterior.predict(query)
        truth=evaluate(pde,case.tensor(),position[0],None if pde=="poisson" else time[0],quantity="v" if quantity else "u")
        draws=bank.evaluate(query)[:,:,0].numpy()
        x=axis.numpy();mean=pred.mean.flatten().numpy();sd=pred.latent_variance.sqrt().flatten().numpy()
        ax.fill_between(x,mean-2*sd,mean+2*sd,color="#2378a0",alpha=.15,label="pointwise latent mean ± 2 SD")
        for i,draw in enumerate(draws):ax.plot(x,draw,color="#519bbc",alpha=.4,lw=.8,label="coherent posterior draws" if i==0 else None)
        ax.plot(x,mean,color="#176b90",lw=2,label="analytic posterior mean")
        ax.plot(x,truth[:,0].numpy(),color="#b13a2f",ls="--",label="truth")
        ax.set(title=f"{pde.title()} {'v' if quantity else 'u'}",xlabel="x, y=0.61" if pde=="poisson" else "physical time, position=(0.37,0.61)")
        ax.legend(fontsize=7)
    fig.suptitle("One coefficient draw per complete field — wave u/v reuse the same coefficient samples")
    return fig
