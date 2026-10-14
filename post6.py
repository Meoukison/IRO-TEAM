import pandas as pd,numpy as np,pickle
from scipy.stats import gaussian_kde
import post3
from post3 import GRID,G,logf
from post4 import add_offeq,seg
from post5 import prior
EDG=[0,20,40,200]
def fit_dens(d,bw=0.025):
    s,m=seg(d); T=d.target.clip(lower=1).values; q=m/T; L=[]
    for lo,hi in zip(EDG[:-1],EDG[1:]):
        Lb=[]
        for k in range(5):
            x=q[(s==k)&(T>max(lo,4))&(T<=hi)]
            if len(x)<200: x=q[(s==k)&(T>8)]
            Lb.append(np.log(gaussian_kde(x,bw_method=bw/x.std())(GRID)+1e-4))
        L.append(np.array(Lb))
    return L
def lf(L,k,q,T):
    out=logf(L[0],k,q)
    for i in (1,2): out=np.where(T>EDG[i],logf(L[i],k,q),out)
    return out
def feats(d,L,pr):
    s,m=seg(d); n=len(d); out={k:np.full(n,np.nan) for k in ['p6','p6_sd','p6_b1']}
    for pid,idx in d.groupby('patient_id',sort=False).indices.items():
        k=s[idx]>=0; t=d.age.values[idx]; tc=t-t.mean(); lo,hi=tc.min(),tc.max()
        ok=(G[1]+2*G[2]*lo>=0)&(G[1]+2*G[2]*hi>=0); g0,g1,g2=G[0][ok],G[1][ok],G[2][ok]
        ll=-0.5*(g1-pr['b1'][pid])**2/pr['b1_sd']**2-0.5*g2**2/0.15**2-0.5*(g0-pr['b0'][pid])**2/pr['b0_sd']**2
        T=np.clip(g0[:,None]+g1[:,None]*tc[None,:]+g2[:,None]*tc[None,:]**2,0.5,None)
        for j in np.where(k)[0]: ll+=lf(L,s[idx][j],m[idx][j]/T[:,j],T[:,j])-np.log(T[:,j])
        w=np.exp(ll-ll.max()); w/=w.sum(); mu=w@T
        out['p6'][idx]=mu; out['p6_sd'][idx]=np.sqrt(np.clip(w@T**2-mu**2,0,None)); out['p6_b1'][idx]=w@g1
    return pd.DataFrame(out,index=d.index)
if __name__=='__main__':
    tr=pd.read_pickle('tr2.pkl'); te=pd.read_pickle('te2.pkl'); folds=pickle.load(open('folds.pkl','rb')); y=tr.target.values
    add_offeq(tr,te)
    E=pd.DataFrame(index=tr.index,columns=['p6','p6_sd','p6_b1'],dtype=float)
    for a,b in folds:
        A=tr.iloc[a]; E.iloc[b]=feats(tr.iloc[b],fit_dens(A),prior(A,tr.iloc[b])).values
    Et=feats(te,fit_dens(tr),prior(tr,te))
    print('p6 OOF rmse %.4f'%np.sqrt(np.mean((E.p6.values-y)**2)),flush=True)
    E['Index']=tr.Index.values; Et['Index']=te.Index.values; E.to_pickle('p6_tr.pkl'); Et.to_pickle('p6_te.pkl')
