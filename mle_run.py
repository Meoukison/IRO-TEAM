import pandas as pd,numpy as np,pickle
from mle import *
import mle
tr=pd.read_pickle('tr2.pkl'); te=pd.read_pickle('te2.pkl'); folds=pickle.load(open('folds.pkl','rb'))
def run_off(d,L):
    s,m=seg(d); out=np.full(len(d),np.nan); nobs=np.zeros(len(d))
    for pid,idx in d.groupby('patient_id',sort=False).indices.items():
        k=(s[idx]==0)|(s[idx]==1)
        if k.sum()==0: continue
        t=d.age.values[idx]; tk=t[k]
        _,b=fit_patient(tk,m[idx][k],s[idx][k],L)
        out[idx]=np.vander(t-tk.mean(),3,increasing=True)@b; nobs[idx]=k.sum()
    return out,nobs
oof=np.full(len(tr),np.nan)
for a,b in folds:
    L=fit_dens(tr.iloc[a]); oof[b],_=run_off(tr.iloc[b],L)
tr['mle_off']=oof; te['mle_off'],_=run_off(te,fit_dens(tr))
y=tr.target.values; m=~np.isnan(oof)
print('mle_off coverage %.3f RMSE %.4f'%(m.mean(),np.sqrt(np.mean((oof[m]-y[m])**2))))
# t4=pd.read_pickle('tr4.pkl'); print('m2q same rows %.4f'%np.sqrt(np.mean((t4.m2q.values[m]-y[m])**2)))
tr[['Index','mle_off']].to_pickle('mle_tr.pkl'); te[['Index','mle_off']].to_pickle('mle_te.pkl')
