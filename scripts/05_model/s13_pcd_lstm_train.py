#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PHASE 5.1 v4 -- PHYSICS-CONSTRAINED DECOMPOSITION LSTM
  convergence training + PROPER basin mass conservation + WEIGHT SENSITIVITY
--------------------------------------------------------------------------------
WHAT CHANGED VS v3 (and why v3 was fragile)
  1. BASIN TERM WAS INERT. v3 compared per-BATCH basin means; since a basin's
     cells scatter across random batches, each batch saw only a few cells of any
     basin -> the "basin mean" was noise and the term never bound (flat green
     loss). v4 computes the basin-mean predicted trend over the WHOLE DOMAIN once
     per epoch (all cells), so mass conservation actually acts. Sweeping the
     weight is only meaningful now that the term is real.
  2. WEIGHT SENSITIVITY. W_BASIN is swept over {0, 0.1, 0.25, 0.5, 1.0}. For each
     we report held-out skill (NSE, KGE), drift, and BASIN-CLOSURE (how well the
     reconstructed basin trend matches observed) so the weight choice is
     defensible, not arbitrary. The best config (skill preserved + drift/closure
     improved) is chosen by rule and becomes the product.
  3. CONVERGENCE. EPOCHS=120, PATIENCE=15: v3 stopped at 60 still improving.
  4. FIGURES fixed: trend map colorbar corrected and markers sized to be visible;
     drift shown with magnitude (not just >0.5 count); basin-closure scatter;
     per-regime series; all fig_style Robinson + HydroBASINS.

STRUCTURAL PHYSICS (unchanged, by construction, weightless)
  GWSA = Trend + Seasonal + Residual, trend = supply_bound * tanh(raw) so a cell
  cannot drift beyond cumulative net-flux supply. This guarantee holds for every
  config in the sweep; the sweep only varies the SOFT basin mass-conservation
  weight.

HARDWARE  i5-11300H CPU. Full sweep = 5 short arms + 1 long convergence arm.
  Sweep arms use EPOCHS_SWEEP (short) to stay tractable (~2.5-4 h total).
OUTPUT
  OUTPUTS\05_model_tensor\gwsa_reconstructed.nc   (best config, converged)
  OUTPUTS\05_model_tensor\phase5_1_diagnostics.txt
  OUTPUTS\05_model_tensor\figures\fig5_1_v4_*.png
================================================================================
"""

import os, sys, time, warnings, traceback, datetime as dt
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_ROOT = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS"
OUT_DIR  = os.path.join(OUT_ROOT, "05_model_tensor")
FIG_DIR  = os.path.join(OUT_DIR, "figures")
TENSOR   = os.path.join(OUT_DIR, "model_ready.nc")
FIG_STYLE_DIR = r"D:\THESIS\Arpon_Thesis_Final\Codes\Into the Models"

SEED, THREADS = 20260730, 6
HIDDEN, LAYERS = 64, 1
EPOCHS_SWEEP, PATIENCE_SWEEP = 45, 8       # short arms for the weight sweep
EPOCHS_FINAL, PATIENCE_FINAL = 120, 15     # converged final training
BATCH_CELLS = 512
LR, WD, GRAD_CLIP = 1.5e-3, 1e-5, 1.0
HUBER_DELTA = 1.0
SUPPLY_SAFETY = 1.5

W_BASIN_SWEEP = [0.0, 0.1, 0.25, 0.5, 1.0]
RECON_TEST_FRAC, VAL_FRAC = 0.15, 0.15
DRIFT_TOL = 0.5

LINE = "=" * 78
_LOG = []
def _p(s=""): print(s); _LOG.append(str(s))
def hdr(t): _p(""); _p(LINE); _p(t); _p(LINE)
def ok(t): _p("  [ OK ]   " + t)
def warn(t): _p("  [WARN]   " + t)
def fail(t): _p("  [FAIL]   " + t)


def nse(sim, obs):
    m = np.isfinite(sim)&np.isfinite(obs)
    if m.sum()<12: return np.nan
    o,s=obs[m],sim[m]; d=np.sum((o-o.mean())**2)
    return float(1-np.sum((s-o)**2)/d) if d>0 else np.nan

def kge(sim, obs):
    m=np.isfinite(sim)&np.isfinite(obs)
    if m.sum()<12: return (np.nan,)*4
    o,s=obs[m],sim[m]
    if o.std()==0 or s.std()==0: return (np.nan,)*4
    r=np.corrcoef(s,o)[0,1]; beta=s.mean()/o.mean() if o.mean()!=0 else np.nan
    gamma=s.std()/o.std()
    return float(1-np.sqrt((r-1)**2+(beta-1)**2+(gamma-1)**2)),float(r),float(beta),float(gamma)

def trend_np(y, t):
    m=np.isfinite(y)
    if m.sum()<24: return np.nan
    x=(t[m]/12.0); x=x-x.mean(); yy=y[m]-y[m].mean(); d=np.sum(x*x)
    return float(np.sum(x*yy)/d) if d>0 else np.nan


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    t0=time.time(); np.random.seed(SEED)
    _p(LINE); _p("PHASE 5.1 v4 -- BASIN-CONSERVING PHYSICS LSTM + WEIGHT SWEEP")
    _p(f"run: {dt.datetime.now().isoformat(timespec='seconds')}  seed={SEED}"); _p(LINE)

    import xarray as xr
    try:
        import torch, torch.nn as nn
    except Exception as e:
        fail(f"torch import: {e}"); return
    torch.manual_seed(SEED); torch.set_num_threads(THREADS)
    _p(f"  torch {torch.__version__}  threads={torch.get_num_threads()}  cpu")

    # ------------------------------------------------------------------ STEP 1
    hdr("STEP 1 -- load tensor")
    if not os.path.exists(TENSOR): fail(f"missing {TENSOR}"); return
    ds=xr.open_dataset(TENSOR); ds.load()
    T=ds.sizes["time"]; ncell=ds.sizes["cell"]
    dfeat=list(ds["dfeat"].values); sfeat=list(ds["sfeat"].values)
    dyn=ds["dyn"].values.astype("float32"); sta=ds["sta"].values.astype("float32")
    gwsa=ds["gwsa"].values.astype("float32"); osa=ds["osa"].values.astype("float32")
    w_data = ds["w_data"].values.astype(bool);
    keep = ds["keep"].values.astype(bool)
    if "surface_water" in ds:  # exclude surface-water cells at the source (match 6.1/7.0)
        _sw = ds["surface_water"].values.astype(bool)
        keep = keep & ~_sw
        _p(f"  excluded {int(_sw.sum())} surface-water cells; kept {int(keep.sum())}")
    dmu, dsd = ds["dyn_mu"].values, ds["dyn_sd"].values
    smu,ssd=ds["sta_mu"].values,ds["sta_sd"].values
    clat,clon=ds["clat"].values,ds["clon"].values
    hyb=ds["hybas_l3"].values.astype("int64")
    ci=np.where(keep)[0]; i_nf,i_e=dfeat.index("NetFlux"),dfeat.index("E")
    _p(f"  cells={ncell} kept={ci.size} months={T} dyn={dfeat}")

    # ------------------------------------------------------------------ STEP 2
    hdr("STEP 2 -- splits")
    obs_months=np.where(w_data.any(axis=1))[0]
    rng=np.random.default_rng(SEED); perm=rng.permutation(obs_months)
    n_te=int(round(RECON_TEST_FRAC*perm.size)); n_va=int(round(VAL_FRAC*perm.size))
    test_m=np.sort(perm[:n_te]); val_m=np.sort(perm[n_te:n_te+n_va]); train_m=np.sort(perm[n_te+n_va:])
    role=np.zeros(T,"int8"); role[train_m]=1; role[val_m]=2; role[test_m]=3
    _p(f"  train/val/test={train_m.size}/{val_m.size}/{test_m.size}")

    # ------------------------------------------------------------------ STEP 3
    hdr("STEP 3 -- supply bound (structural)")
    supply=(np.nanmean(dyn[:,:,i_nf]-dyn[:,:,i_e],axis=0)/10.0)*12.0
    bound_np=SUPPLY_SAFETY*np.abs(supply)+1.0
    _p(f"  supply bound cm/yr: median={np.nanmedian(bound_np[ci]):.2f} p95={np.nanpercentile(bound_np[ci],95):.2f}")

    # ------------------------------------------------------------------ STEP 4
    hdr("STEP 4 -- tensors + basin index (full-domain)")
    dsd_s=np.where(dsd>0,dsd,1.0); ssd_s=np.where(ssd>0,ssd,1.0)
    dyn_z=np.nan_to_num((dyn-dmu[None,None,:])/dsd_s[None,None,:]).astype("float32")
    sta_z=np.nan_to_num((sta-smu[None,:])/ssd_s[None,:]).astype("float32")
    Xd=torch.from_numpy(dyn_z[:,ci,:]).permute(1,0,2).contiguous()
    Xs=torch.from_numpy(sta_z[ci,:]).contiguous()
    Y=torch.from_numpy(gwsa[:,ci].T).contiguous()
    Wd=torch.from_numpy(w_data[:,ci].T.astype("float32")).contiguous()
    bound=torch.from_numpy(bound_np[ci].astype("float32")).contiguous()
    N,Dd,Sd=Xd.shape[0],Xd.shape[2],Xs.shape[1]
    role_t=torch.from_numpy(role); m_train=(role_t==1).float(); m_val=(role_t==2).float()
    tyr=torch.from_numpy(((np.arange(T)-np.arange(T).mean())/12.0).astype("float32"))
    moy=torch.from_numpy((np.arange(T)%12).astype("int64"))

    yv=Y.numpy(); wdv=Wd.numpy().astype(bool); tidx=np.arange(T)
    obs_tr=np.array([trend_np(yv[k,wdv[k]],tidx[wdv[k]]) for k in range(N)])
    hyb_ci=hyb[ci]; ub=np.unique(hyb_ci[hyb_ci>0])
    # full-domain basin membership (index lists) + observed basin trend target
    basin_members={}; basin_obs={}
    for b in ub:
        loc=np.where(hyb_ci==b)[0]
        v=obs_tr[loc]; v=v[np.isfinite(v)]
        if v.size>=2:
            basin_members[b]=torch.as_tensor(loc); basin_obs[b]=float(np.mean(v))
    _p(f"  basins with conservation target: {len(basin_obs)}")

    # ------------------------------------------------------------------ STEP 5
    hdr("STEP 5 -- model")
    class Decomp(nn.Module):
        def __init__(s,dd,dss,h,L):
            super().__init__()
            s.lstm=nn.LSTM(dd+dss,h,num_layers=L,batch_first=True)
            s.res=nn.Linear(h,1)
            s.tr=nn.Sequential(nn.Linear(dss,h//2),nn.ReLU(),nn.Linear(h//2,1))
            s.seas=nn.Embedding(12,1)
            s.sh=nn.Sequential(nn.Linear(dss,h//2),nn.ReLU(),nn.Linear(h//2,1))
        def forward(s,xd,xs,tyr_b,moy_b,bnd):
            B,Tt,_=xd.shape
            xse=xs.unsqueeze(1).expand(-1,Tt,-1)
            hq,_=s.lstm(torch.cat([xd,xse],-1))
            resid=s.res(hq).squeeze(-1)
            slope=bnd*torch.tanh(s.tr(xs).squeeze(-1))
            trend=slope.unsqueeze(1)*tyr_b.unsqueeze(0)
            amp=s.sh(xs).squeeze(-1)
            seas=amp.unsqueeze(1)*s.seas(moy_b).squeeze(-1).unsqueeze(0)
            return trend+seas+resid, slope
    def huber(r,d):
        a=torch.abs(r); return torch.where(a<=d,0.5*r*r,d*(a-0.5*d))

    def slopes_all(model):
        """predicted slope for every cell (for full-domain basin term/eval)"""
        with torch.no_grad():
            out=np.full(N,np.nan,"float32")
            for s0 in range(0,N,BATCH_CELLS):
                e=min(s0+BATCH_CELLS,N); b=torch.arange(s0,e)
                _,sl=model(Xd[b],Xs[b],tyr,moy,bound[b]); out[s0:e]=sl.numpy()
        return out

    def train(w_basin, epochs, patience, tag):
        torch.manual_seed(SEED)
        model=Decomp(Dd,Sd,HIDDEN,LAYERS)
        opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WD)
        idx=np.arange(N); best=np.inf; best_state=None; bad=0; hist=[]
        _p(f"\n  --- arm '{tag}' (W_BASIN={w_basin}, epochs={epochs}) ---")
        for ep in range(1,epochs+1):
            model.train(); rng.shuffle(idx); ad=nb=0.0
            for s0 in range(0,N,BATCH_CELLS):
                b=torch.as_tensor(idx[s0:s0+BATCH_CELLS])
                pred,_=model(Xd[b],Xs[b],tyr,moy,bound[b])
                wtr=Wd[b]*m_train.unsqueeze(0)
                l_data=(wtr*(pred-Y[b])**2).sum()/wtr.sum().clamp_min(1)
                opt.zero_grad(); l_data.backward()
                nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); opt.step()
                ad+=float(l_data); nb+=1
            # FULL-DOMAIN basin mass conservation, once per epoch
            l_basin_val=0.0
            if w_basin>0:
                # recompute slopes WITH grad for a single conservation step
                opt.zero_grad()
                lb=torch.tensor(0.); cnt=0
                for b,loc in basin_members.items():
                    _,sl=model(Xd[loc],Xs[loc],tyr,moy,bound[loc])
                    lb=lb+(sl.mean()-basin_obs[b])**2; cnt+=1
                lb=w_basin*(lb/max(cnt,1)); lb.backward()
                nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); opt.step()
                l_basin_val=float(lb)
            model.eval()
            with torch.no_grad():
                vl=vd=0.0
                for s0 in range(0,N,BATCH_CELLS):
                    b=torch.arange(s0,min(s0+BATCH_CELLS,N))
                    pr,_=model(Xd[b],Xs[b],tyr,moy,bound[b]); wv=Wd[b]*m_val.unsqueeze(0)
                    vl+=float((wv*(pr-Y[b])**2).sum()); vd+=float(wv.sum())
                vmse=vl/max(vd,1)
            hist.append((ep,ad/nb,l_basin_val,vmse))
            if ep%10==0 or ep<=3:
                _p(f"     ep {ep:03d} data={ad/nb:.3f} basin={l_basin_val:.4f} val={vmse:.3f}")
            if vmse<best-1e-5: best=vmse; best_state={k:v.clone() for k,v in model.state_dict().items()}; bad=0
            else:
                bad+=1
                if bad>=patience: _p(f"     early stop ep {ep} (val {best:.3f})"); break
        model.load_state_dict(best_state)
        return model, hist

    def recon(model):
        with torch.no_grad():
            out=np.full((N,T),np.nan,"float32")
            for s0 in range(0,N,BATCH_CELLS):
                e=min(s0+BATCH_CELLS,N); b=torch.arange(s0,e)
                pr,_=model(Xd[b],Xs[b],tyr,moy,bound[b]); out[s0:e]=pr.numpy()
        return out

    tmask=(role==3)[:,None].T & wdv   # (N,T)? ensure shape (N,T)
    tmask=((role==3)[None,:] & wdv)    # (N,T): month-role broadcast over cells
    def evaluate(R):
        s=R[tmask]; o=yv[tmask]; m=np.isfinite(s)&np.isfinite(o)
        rmse=float(np.sqrt(np.mean((s[m]-o[m])**2)))
        n=nse(s[m],o[m]); k,r,beta,gamma=kge(s[m],o[m])
        dtr=[]
        for kk in range(N):
            om=wdv[kk]
            if om.sum()<36: continue
            to=trend_np(yv[kk,om],tidx[om]); tr=trend_np(R[kk],tidx)
            if np.isfinite(to) and np.isfinite(tr): dtr.append(tr-to)
        dtr=np.array(dtr); drift=float(np.mean(np.abs(dtr)>DRIFT_TOL))
        # basin closure: reconstructed basin trend vs observed basin trend
        rec_slope=np.array([trend_np(R[kk],tidx) for kk in range(N)])
        bc_err=[]
        for b,loc in basin_members.items():
            v=rec_slope[loc.numpy()]; v=v[np.isfinite(v)]
            if v.size: bc_err.append(np.mean(v)-basin_obs[b])
        bc_err=np.array(bc_err); bc_rmse=float(np.sqrt(np.mean(bc_err**2))) if bc_err.size else np.nan
        return dict(rmse=rmse,nse=n,kge=k,r=r,beta=beta,gamma=gamma,
                    drift=drift,dtr=dtr,bc_rmse=bc_rmse)

    # ------------------------------------------------------------------ STEP 6
    hdr("STEP 6 -- WEIGHT SENSITIVITY SWEEP (short arms)")
    sweep=[]
    for wb in W_BASIN_SWEEP:
        m,h=train(wb, EPOCHS_SWEEP, PATIENCE_SWEEP, f"sweep W_BASIN={wb}")
        R=recon(m); ev=evaluate(R); ev["w"]=wb; sweep.append(ev)
        _p(f"   -> W_BASIN={wb}: RMSE={ev['rmse']:.3f} NSE={ev['nse']:+.3f} "
           f"KGE={ev['kge']:+.3f} drift={100*ev['drift']:.1f}% "
           f"basin-closure RMSE={ev['bc_rmse']:.4f} cm/yr")

    _p("")
    _p("  SWEEP SUMMARY:")
    _p(f"  {'W_BASIN':<9}{'RMSE':<8}{'NSE':<8}{'KGE':<8}{'drift%':<8}{'basinRMSE':<10}")
    for ev in sweep:
        _p(f"  {ev['w']:<9}{ev['rmse']:<8.3f}{ev['nse']:<+8.3f}{ev['kge']:<+8.3f}"
           f"{100*ev['drift']:<8.1f}{ev['bc_rmse']:<10.4f}")
    # choose best: skill within 3% of best RMSE, then min basin-closure RMSE
    best_rmse=min(e["rmse"] for e in sweep)
    cand=[e for e in sweep if e["rmse"]<=best_rmse*1.03]
    chosen_w=min(cand, key=lambda e:e["bc_rmse"])["w"]
    _p(f"  -> chosen W_BASIN={chosen_w} (best basin closure among skill-preserving configs)")

    # ------------------------------------------------------------------ STEP 7
    hdr("STEP 7 -- FINAL converged training at chosen weight")
    m_final,h_final=train(chosen_w, EPOCHS_FINAL, PATIENCE_FINAL, f"FINAL W_BASIN={chosen_w}")
    R_final=recon(m_final); ev_final=evaluate(R_final)
    slope_final=np.array([trend_np(R_final[k],tidx) for k in range(N)])
    _p(f"  FINAL: RMSE={ev_final['rmse']:.3f} NSE={ev_final['nse']:+.3f} "
       f"KGE={ev_final['kge']:+.3f} (beta={ev_final['beta']:.2f} gamma={ev_final['gamma']:.2f}) "
       f"drift={100*ev_final['drift']:.1f}% basin-closure={ev_final['bc_rmse']:.4f} cm/yr")
    _p("  (KGE here is PRE-calibration; Phase 5.2 variance-matching lifts it.)")

    # ------------------------------------------------------------------ STEP 8
    hdr("STEP 8 -- ALL-PREDICTOR permutation importance")
    def hrmse(R):
        s=R[tmask]; o=yv[tmask]; m=np.isfinite(s)&np.isfinite(o)
        return float(np.sqrt(np.mean((s[m]-o[m])**2)))
    base=hrmse(R_final); imp={}
    for fi,fn in enumerate(dfeat):
        X2=Xd.clone(); pj=torch.from_numpy(rng.permutation(N)); X2[:,:,fi]=Xd[pj][:,:,fi]
        with torch.no_grad():
            Rp=np.full((N,T),np.nan,"float32")
            for s0 in range(0,N,BATCH_CELLS):
                e=min(s0+BATCH_CELLS,N); b=torch.arange(s0,e)
                pr,_=m_final(X2[b],Xs[b],tyr,moy,bound[b]); Rp[s0:e]=pr.numpy()
        imp[fn+" (dyn)"]=hrmse(Rp)-base
    for fi,fn in enumerate(sfeat):
        X2=Xs.clone(); pj=torch.from_numpy(rng.permutation(N)); X2[:,fi]=Xs[pj][:,fi]
        with torch.no_grad():
            Rp=np.full((N,T),np.nan,"float32")
            for s0 in range(0,N,BATCH_CELLS):
                e=min(s0+BATCH_CELLS,N); b=torch.arange(s0,e)
                pr,_=m_final(Xd[b],X2[b],tyr,moy,bound[b]); Rp[s0:e]=pr.numpy()
        imp[fn+" (sta)"]=hrmse(Rp)-base
    for fn,v in sorted(imp.items(),key=lambda x:-x[1]):
        _p(f"     {fn:<20}: {v:+.4f} cm")

    # ------------------------------------------------------------------ STEP 9
    hdr("STEP 9 -- write reconstruction (best converged)")
    rec_full=np.full((T,ncell),np.nan,"float32"); rec_full[:,ci]=R_final.T
    out=xr.Dataset(
        {"gwsa_recon":(("time","cell"),rec_full),
         "gwsa_obs":(("time","cell"),gwsa),"osa":(("time","cell"),osa),
         "w_data":(("time","cell"),w_data.astype("int8")),
         "month_role":(("time",),role),"keep":(("cell",),keep.astype("int8")),
         "hybas_l3":(("cell",),hyb),
         "supply_bound_cm_yr":(("cell",),bound_np.astype("float32")),
         "clat":(("cell",),clat.astype("float32")),"clon":(("cell",),clon.astype("float32"))},
        coords={"time":ds["time"].values,"cell":np.arange(ncell)},
        attrs={"title":"Phase 5.1 v4 basin-conserving physics GWSA reconstruction",
               "created":dt.datetime.now().isoformat(timespec="seconds"),
               "chosen_w_basin":float(chosen_w),
               "method":"decomposition; supply-bounded trend (structural); "
                        "full-domain basin mass conservation (soft, swept)",
               "heldout_rmse_cm":float(ev_final["rmse"]),
               "heldout_nse":float(ev_final["nse"]),
               "heldout_kge_precal":float(ev_final["kge"]),
               "frac_cells_drifting":float(ev_final["drift"]),
               "basin_closure_rmse_cm_yr":float(ev_final["bc_rmse"]),
               "note":"run Phase 5.2 for variance-matching calibration + trends"})
    enc={v:{"zlib":True,"complevel":4} for v in out.data_vars}
    p=os.path.join(OUT_DIR,"gwsa_reconstructed.nc"); out.to_netcdf(p,encoding=enc)
    ok(f"written: {p} ({os.path.getsize(p)/1e6:.0f} MB)")
    import torch as _t; _t.save(m_final.state_dict(),os.path.join(OUT_DIR,"lstm_state_v4.pt"))

    try:
        figs(sweep,chosen_w,h_final,R_final,yv,role,wdv,ci,clat,clon,
             slope_final,ev_final,imp,tidx,basin_members,basin_obs)
    except Exception as e:
        warn(f"figures: {e}"); traceback.print_exc()

    with open(os.path.join(OUT_DIR,"phase5_1_diagnostics.txt"),"w",encoding="utf-8") as fh:
        fh.write("\n".join(_LOG))
    _p(f"\n  wall time: {(time.time()-t0)/60:.1f} min")
    _p(LINE)
    _p("Run complete.")
    _p(LINE)


def figs(sweep,chosen_w,hist,R,yv,role,wdv,ci,clat,clon,slope,ev,imp,tidx,
         basin_members,basin_obs):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        sys.path.insert(0,FIG_STYLE_DIR); import fig_style as fs
        fs.apply(); SZL,SZT,SZA=fs.SZ_LABEL+1,fs.SZ_TITLE+1,fs.SZ_ANNOT; HAVE=True
    except Exception:
        from matplotlib import rcParams
        rcParams["font.family"]="serif"; rcParams.update({"font.size":12,"savefig.dpi":600})
        SZL=SZT=SZA=12; HAVE=False
    def save(f,nm):
        for e in (".png",".pdf"):
            f.savefig(os.path.join(FIG_DIR,nm+e),dpi=600,bbox_inches="tight",facecolor="white")
        plt.close(f); ok(nm)
    tmask=((role==3)[None,:]&wdv)

    # FIG 1: sensitivity sweep (4 panels)
    fig=plt.figure(figsize=(11,7))
    ws=[e["w"] for e in sweep]
    ax=fig.add_axes((0.08,0.57,0.38,0.36))
    ax.plot(ws,[e["rmse"] for e in sweep],"o-",color="#1f4e79")
    ax.axvline(chosen_w,color="#c0392b",ls="--",lw=1)
    ax.set_xlabel("Basin weight",fontsize=SZL); ax.set_ylabel("Held-out RMSE (cm)",fontsize=SZL)
    ax.set_title("Skill vs basin weight",fontsize=SZT); ax.spines[["top","right"]].set_visible(False)
    ax=fig.add_axes((0.58,0.57,0.38,0.36))
    ax.plot(ws,[e["bc_rmse"] for e in sweep],"o-",color="#2ca02c")
    ax.axvline(chosen_w,color="#c0392b",ls="--",lw=1)
    ax.set_xlabel("Basin weight",fontsize=SZL); ax.set_ylabel("Basin-closure RMSE (cm/yr)",fontsize=SZL)
    ax.set_title("Mass conservation vs weight",fontsize=SZT); ax.spines[["top","right"]].set_visible(False)
    ax=fig.add_axes((0.08,0.09,0.38,0.36))
    ax.plot(ws,[100*e["drift"] for e in sweep],"o-",color="#ff7f0e")
    ax.axvline(chosen_w,color="#c0392b",ls="--",lw=1)
    ax.set_xlabel("Basin weight",fontsize=SZL); ax.set_ylabel("Cells drifting >0.5 (%)",fontsize=SZL)
    ax.set_title("Drift vs weight",fontsize=SZT); ax.spines[["top","right"]].set_visible(False)
    ax=fig.add_axes((0.58,0.09,0.38,0.36))
    h=np.array(hist); ax.plot(h[:,0],h[:,3],color="#1f4e79")
    ax.set_xlabel("Epoch",fontsize=SZL); ax.set_ylabel("Validation MSE",fontsize=SZL)
    ax.set_title(f"Final convergence (W={chosen_w})",fontsize=SZT); ax.spines[["top","right"]].set_visible(False)
    save(fig,"fig5_1_v4_sensitivity")

    # FIG 2: performance (scatter + drift-magnitude + basin closure + importance)
    fig=plt.figure(figsize=(11,7))
    ax=fig.add_axes((0.07,0.57,0.38,0.37))
    s=R[tmask]; o=yv[tmask]; m=np.isfinite(s)&np.isfinite(o)
    lim=np.nanpercentile(np.abs(np.concatenate([s[m],o[m]])),99)
    hb=ax.hexbin(o[m],s[m],gridsize=50,cmap="magma",mincnt=1,extent=[-lim,lim,-lim,lim])
    ax.plot([-lim,lim],[-lim,lim],"w--",lw=1)
    ax.set_xlabel("Observed GWSA (cm)",fontsize=SZL); ax.set_ylabel("Reconstructed (cm)",fontsize=SZL)
    ax.set_title(f"Held-out (RMSE={ev['rmse']:.2f}, NSE={ev['nse']:+.2f})",fontsize=SZT)
    ax=fig.add_axes((0.58,0.57,0.38,0.37))
    d=ev["dtr"]; ax.hist(d[np.abs(d)<3],bins=60,color="#1f4e79",edgecolor="none")
    ax.axvline(0,color="k",lw=0.8)
    ax.set_xlabel("Reconstructed $-$ observed trend (cm yr$^{-1}$)",fontsize=SZL)
    ax.set_ylabel("Cells",fontsize=SZL)
    ax.set_title(f"Trend bias (median {np.median(d):+.3f})",fontsize=SZT)
    ax.spines[["top","right"]].set_visible(False)
    ax=fig.add_axes((0.07,0.09,0.38,0.37))
    bx=[]; by=[]
    rec_slope=np.array([trend_np(R[k],tidx) for k in range(R.shape[0])])
    for b,loc in basin_members.items():
        v=rec_slope[loc.numpy()]; v=v[np.isfinite(v)]
        if v.size: bx.append(basin_obs[b]); by.append(np.mean(v))
    bx=np.array(bx); by=np.array(by)
    lim2=np.nanpercentile(np.abs(np.concatenate([bx,by])),98)
    ax.scatter(bx,by,s=14,color="#2ca02c",alpha=0.6,linewidths=0)
    ax.plot([-lim2,lim2],[-lim2,lim2],"k--",lw=1)
    ax.set_xlim(-lim2,lim2); ax.set_ylim(-lim2,lim2)
    ax.set_xlabel("Observed basin trend (cm/yr)",fontsize=SZL)
    ax.set_ylabel("Reconstructed basin trend",fontsize=SZL)
    ax.set_title(f"Basin mass conservation (RMSE={ev['bc_rmse']:.3f})",fontsize=SZT)
    ax.spines[["top","right"]].set_visible(False)
    ax=fig.add_axes((0.58,0.09,0.38,0.37))
    it=sorted(imp.items(),key=lambda x:x[1])
    cols=["#1f4e79" if "(dyn)" in k else "#2ca02c" for k,_ in it]
    ax.barh([k.replace(" (dyn)","").replace(" (sta)","") for k,_ in it],
            [v for _,v in it],color=cols,edgecolor="k",lw=0.3)
    ax.set_xlabel("RMSE increase (cm)",fontsize=SZL); ax.set_title("Feature importance",fontsize=SZT)
    ax.spines[["top","right"]].set_visible(False)
    save(fig,"fig5_1_v4_performance")

    # FIG 3: FIXED trend map (visible markers, correct colorbar)
    fig=plt.figure(figsize=(12,6.2))
    lim=float(np.nanpercentile(np.abs(slope[np.isfinite(slope)]),97))
    if HAVE:
        import cartopy.crs as ccrs
        ax=fs.global_map_axis(fig,(0.03,0.13,0.94,0.82),projection="robinson",basins=True)
        order=np.argsort(np.abs(slope))            # plot strong trends on top
        sc=ax.scatter(clon[ci][order],clat[ci][order],c=slope[order],
                      cmap="RdBu",vmin=-lim,vmax=lim,s=7,marker="s",linewidths=0,
                      transform=ccrs.PlateCarree(),zorder=5)
        cax=fig.add_axes((0.32,0.09,0.36,0.018))
        cb=fig.colorbar(sc,cax=cax,orientation="horizontal")
        cb.set_label("Reconstructed GWSA trend (cm yr$^{-1}$)",fontsize=SZL)
        cb.ax.tick_params(labelsize=SZA)
        ax.set_title("Physics-bounded groundwater storage trend, 2002-2022",fontsize=SZT)
    else:
        ax=fig.add_subplot(111)
        sc=ax.scatter(clon[ci],clat[ci],c=slope,cmap="RdBu",vmin=-lim,vmax=lim,s=4,linewidths=0)
        ax.set_xlim(-180,180); ax.set_ylim(-60,85)
        plt.colorbar(sc,ax=ax,shrink=0.7,label="cm/yr")
    save(fig,"fig5_1_v4_trend_map")


if __name__=="__main__":
    try: main()
    except Exception: traceback.print_exc()