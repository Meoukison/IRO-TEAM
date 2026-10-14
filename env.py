import pandas as pd,numpy as np,pickle
from scipy.optimize import minimize
from mle import fit_dens,logf,seg
def fit_lin(t,m,s,L):
    tc=t-t.mean(); A=np.c_[np.ones(len(t)),tc]
    def nll(b):
        T=np.clip(A@b,1.0,None); return -(np.sum(logf(L,s,m/T)-np.log(T))-0.5*(b[1]-2.5)**2/1.5**2)
    st=np.array([max(np.percentile(m,80),2),2.5]); r=min((minimize(nll,x,method='Nelder-Mead',options=dict(xatol=1e-3,fatol=1e-4)) for x in (st,st*[1.2,1],st*[0.9,1])),key=lambda r:r.fun)
    return r.x,t.mean()
def env_feats(d,L):
    s,m=seg(d); n=len(d); out={k:np.full(n,np.nan) for k in ['mle_lin','env_q80','env_q95','env_max','roll_max3','roll_max5','env_lin_res_sd']}
    for pid,idx in d.groupby('patient_id',sort=False).indices.items():
        t=d.age.values[idx]; off=d.off.values[idx]; k=~np.isnan(off)
        if k.sum()==0: continue
        b,tm=fit_lin(t[k],off[k],s[idx][k],L); out['mle_lin'][idx]=b[0]+b[1]*(t-tm)
        # LS linear fit w/ slope prior then residual quantile shift
        tk=t[k]-t[k].mean(); A=np.c_[np.ones(k.sum()),tk]; bb=np.linalg.solve(A.T@A+np.diag([1e-6,3.0]),A.T@off[k]+np.array([0,3.0*2.5]))
        res=off[k]-A@bb; base=bb[0]+bb[1]*(t-t[k].mean())
        for q,nm in [(80,'env_q80'),(95,'env_q95'),(100,'env_max')]: out[nm][idx]=base+np.percentile(res,q)
        out['env_lin_res_sd'][idx]=res.std() if k.sum()>1 else np.nan
        # rolling max of slope-detrended off within +-w visits
        det=off-2.5*(t-t.mean())
        for w,nm in [(1,'roll_max3'),(2,'roll_max5')]:
            out[nm][idx]=[np.nanmax(det[max(0,i-w):i+w+1]) if np.isfinite(det[max(0,i-w):i+w+1]).any() else np.nan for i in range(len(idx))]
            out[nm][idx]+=2.5*(t-t.mean())
    return pd.DataFrame(out,index=d.index)
if __name__=='__main__':
    tr=pd.read_pickle('tr2.pkl'); te=pd.read_pickle('te2.pkl'); folds=pickle.load(open('folds.pkl','rb')); y=tr.target.values
    E=pd.DataFrame(index=tr.index,columns=['mle_lin','env_q80','env_q95','env_max','roll_max3','roll_max5','env_lin_res_sd'],dtype=float)
    for a,b in folds:
        E.iloc[b]=env_feats(tr.iloc[b],fit_dens(tr.iloc[a])).values
    Et=env_feats(te,fit_dens(tr))
    for c in E: mm=E[c].notna(); print(c,'rmse vs target %.3f'%np.sqrt(np.mean((E[c][mm]-y[mm])**2)))
    E['Index']=tr.Index.values; Et['Index']=te.Index.values; E.to_pickle('env_tr.pkl'); Et.to_pickle('env_te.pkl')
