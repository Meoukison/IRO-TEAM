import pandas as pd,numpy as np,pickle,sys
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
GRID=np.linspace(-0.2,2.0,2201)
def seg(d):
    # 0 off-untreated, 1 off-treated, 2 on (only when off missing)
    s=np.full(len(d),-1); off=d.off.notna().values; tre=d.ledd.notna().values
    s[off&~tre]=0; s[off&tre]=1; s[~off&d.on.notna().values]=2
    m=np.where(off,d.off.values,d.on.values)
    return s,m
def fit_dens(d,bw=0.02):
    s,m=seg(d); q=m/d.target.clip(lower=1).values; ok=d.target.values>8
    L=[]
    for k in range(3):
        kde=gaussian_kde(q[(s==k)&ok],bw_method=bw/q[(s==k)&ok].std()); p=kde(GRID)+1e-4; L.append(np.log(p))
    return np.array(L)
def logf(L,k,q):
    qc=np.clip(q,GRID[0],GRID[-1]); pos=(qc-GRID[0])/(GRID[1]-GRID[0]); i=np.minimum(pos.astype(int),len(GRID)-2); f=pos-i
    v=L[k,i]*(1-f)+L[k,i+1]*f
    return v-50*np.clip(q-GRID[-1],0,None)
def fit_patient(t,m,s,L,pm=(None,2.5,0.0),pv=(None,1.5**2,0.3**2)):
    tc=t-t.mean(); A=np.vander(tc,3,increasing=True)
    def nll(b):
        T=np.clip(A@b,1.0,None); q=m/T
        ll=np.sum(logf(L,s,q)-np.log(T))
        return -(ll-0.5*(b[1]-pm[1])**2/pv[1]-0.5*(b[2]-pm[2])**2/pv[2])
    b0=np.array([max(np.percentile(m,80),2),2.5,0.0])
    best=None
    for start in (b0, b0*np.array([1.15,1,1]), b0*np.array([0.9,1,1])):
        r=minimize(nll,start,method='Nelder-Mead',options=dict(xatol=1e-3,fatol=1e-4,maxiter=2000))
        if best is None or r.fun<best.fun: best=r
    return A@best.x, best.x
def run(d,L):
    s,m=seg(d); out=np.full(len(d),np.nan); B={}
    for pid,idx in d.groupby('patient_id',sort=False).indices.items():
        k=s[idx]>=0
        if k.sum()==0: continue
        T,b=fit_patient(d.age.values[idx][k] if False else d.age.values[idx],np.where(k,m[idx],np.nan),s[idx],L) if False else (None,None)
        t=d.age.values[idx]; tc=t-t.mean(); A=np.vander(tc,3,increasing=True)
        T,b=fit_patient(t[k],m[idx][k],s[idx][k],L)
        # evaluate on all visits with same centering as fit
        tk=t[k]; Ak=np.vander(t-tk.mean(),3,increasing=True); out[idx]=Ak@b; B[pid]=b
    return out
if __name__=='__main__':
    tr=pd.read_pickle('tr2.pkl'); y=tr.target.values
    L=fit_dens(tr)
    pids=tr.patient_id.unique()[:800]; sub=tr[tr.patient_id.isin(pids)]
    import time; t0=time.time(); o=run(sub,L); print('time %.1fs'%(time.time()-t0))
    print('MLE RMSE on 800 pts %.4f'%np.sqrt(np.nanmean((o-sub.target.values)**2)))
    from post import polyproj
    print('ref m2q on same', np.sqrt(np.mean((pd.read_pickle('tr4.pkl').set_index('Index').loc[sub.Index,'m2q'].values-sub.target.values)**2)))
