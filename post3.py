import pandas as pd,numpy as np,pickle
from scipy.stats import gaussian_kde
GRID=np.linspace(-0.2,2.0,1101)
def seg(d):
    off=d.off.notna().values; tre=d.ledd.notna().values; first=(d.groupby('patient_id').cumcount()==0).values
    tsion=d.time_since_intake_on.notna().values
    s=np.full(len(d),-1)
    s[off&~tre&first]=0; s[off&~tre&~first]=1; s[off&tre]=2
    s[~off&tsion]=3; s[~off&~tsion]=4
    return s,np.where(off,d.off.values,d.on.values)
def fit_dens(d,bw=0.025):
    s,m=seg(d); q=m/d.target.clip(lower=1).values; ok=d.target.values>8; L=[]
    for k in range(5):
        x=q[(s==k)&ok]; L.append(np.log(gaussian_kde(x,bw_method=bw/x.std())(GRID)+1e-4))
    return np.array(L)
def logf(L,k,q):
    qc=np.clip(q,GRID[0],GRID[-1]); pos=(qc-GRID[0])/(GRID[1]-GRID[0]); i=np.minimum(pos.astype(int),len(GRID)-2); f=pos-i
    return L[k,i]*(1-f)+L[k,i+1]*f-50*np.clip(q-GRID[-1],0,None)
B0=np.arange(1,126,1.0); B1=np.arange(-0.5,8.01,0.5); B2=np.array([-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3])
G=np.array(np.meshgrid(B0,B1,B2,indexing='ij')).reshape(3,-1)
LP=-0.5*(G[1]-2.5)**2/1.5**2-0.5*G[2]**2/0.15**2
def feats(d,L):
    s,m=seg(d); n=len(d); out={k:np.full(n,np.nan) for k in ['p3','p3_sd','p3_b1','p3_b2']}
    for pid,idx in d.groupby('patient_id',sort=False).indices.items():
        k=s[idx]>=0; t=d.age.values[idx]; tm=t.mean(); tc=t-tm
        lo,hi=tc.min(),tc.max()
        ok=(G[1]+2*G[2]*lo>=0)&(G[1]+2*G[2]*hi>=0)
        g0,g1,g2=G[0][ok],G[1][ok],G[2][ok]; ll=LP[ok].copy()
        T=np.clip(g0[:,None]+g1[:,None]*tc[None,:]+g2[:,None]*tc[None,:]**2,0.5,None)
        for j in np.where(k)[0]: ll+=logf(L,s[idx][j],m[idx][j]/T[:,j])-np.log(T[:,j])
        w=np.exp(ll-ll.max()); w/=w.sum(); mu=w@T
        out['p3'][idx]=mu; out['p3_sd'][idx]=np.sqrt(np.clip(w@T**2-mu**2,0,None)); out['p3_b1'][idx]=w@g1; out['p3_b2'][idx]=w@g2
    return pd.DataFrame(out,index=d.index)
if __name__=='__main__':
    tr=pd.read_pickle('tr2.pkl'); te=pd.read_pickle('te2.pkl'); folds=pickle.load(open('folds.pkl','rb')); y=tr.target.values
    import time; t0=time.time()
    sub=tr[tr.patient_id.isin(tr.patient_id.unique()[:300])]; f=feats(sub,fit_dens(tr)); print('300 pts %.1fs rmse %.3f'%(time.time()-t0,np.sqrt(np.mean((f.p3-sub.target)**2))),flush=True)
    E=pd.DataFrame(index=tr.index,columns=['p3','p3_sd','p3_b1','p3_b2'],dtype=float)
    for a,b in folds: E.iloc[b]=feats(tr.iloc[b],fit_dens(tr.iloc[a])).values
    Et=feats(te,fit_dens(tr))
    print('p3 OOF rmse %.4f'%np.sqrt(np.mean((E.p3.values-y)**2)))
    E['Index']=tr.Index.values; Et['Index']=te.Index.values; E.to_pickle('p3_tr.pkl'); Et.to_pickle('p3_te.pkl')
