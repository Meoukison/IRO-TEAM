import pandas as pd,numpy as np,pickle
from mle import fit_dens,logf,seg
B0=np.arange(0.5,125,0.5); B1=np.arange(-3,9.01,0.25)
G0,G1=np.meshgrid(B0,B1,indexing='ij'); G0=G0.ravel(); G1=G1.ravel()
LP1=-0.5*(G1-2.5)**2/1.5**2
def feats(d,L,use_on):
    s,m=seg(d); n=len(d); pm=np.full(n,np.nan); psd=np.full(n,np.nan); pb1=np.full(n,np.nan)
    if not use_on: s=np.where(s==2,-1,s)
    for pid,idx in d.groupby('patient_id',sort=False).indices.items():
        k=s[idx]>=0
        if k.sum()==0: continue
        t=d.age.values[idx]; tm=t[k].mean(); tk=t[k]-tm
        T=np.clip(G0[:,None]+G1[:,None]*tk[None,:],0.5,None)   # grid x obs
        q=m[idx][k][None,:]/T
        ll=LP1.copy()
        for j,sj in enumerate(s[idx][k]): ll+=logf(L,sj,q[:,j])-np.log(T[:,j])
        w=np.exp(ll-ll.max()); w/=w.sum()
        Tall=G0[:,None]+G1[:,None]*(t-tm)[None,:]
        mu=w@Tall; pm[idx]=mu; psd[idx]=np.sqrt(np.clip(w@(Tall**2)-mu**2,0,None)); pb1[idx]=w@G1
    return pm,psd,pb1
if __name__=='__main__':
    tr=pd.read_pickle('tr2.pkl'); te=pd.read_pickle('te2.pkl'); folds=pickle.load(open('folds.pkl','rb')); y=tr.target.values
    out_tr=pd.DataFrame({'Index':tr.Index.values}); out_te=pd.DataFrame({'Index':te.Index.values})
    for use_on,tag in [(False,'g_off'),(True,'g_all')]:
        a_m=np.full(len(tr),np.nan); a_s=a_m.copy(); a_b=a_m.copy()
        for a,b in folds:
            L=fit_dens(tr.iloc[a]); a_m[b],a_s[b],a_b[b]=feats(tr.iloc[b],L,use_on)
        out_tr[tag]=a_m; out_tr[tag+'_sd']=a_s; out_tr[tag+'_b1']=a_b
        out_te[tag],out_te[tag+'_sd'],out_te[tag+'_b1']=feats(te,fit_dens(tr),use_on)
        mm=~np.isnan(a_m); print(tag,'cov %.3f RMSE %.4f'%(mm.mean(),np.sqrt(np.mean((a_m[mm]-y[mm])**2))),flush=True)
    out_tr.to_pickle('grid_tr.pkl'); out_te.to_pickle('grid_te.pkl')
