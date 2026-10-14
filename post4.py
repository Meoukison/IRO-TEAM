import pandas as pd,numpy as np,pickle,lightgbm as lgb
from scipy.stats import gaussian_kde
import post3
from post3 import GRID,logf,G,LP
X=['time_since_intake_on','ledd','dur']
def add_offeq(tr,te,k=2.0):
    # no target used: fit on/off ratio model on all both-visits (train+test)
    a=pd.concat([tr,te]); b=a[a.on.notna()&a.off.notna()&(a.off>5)]
    m=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=15,min_child_samples=100,verbose=-1,n_jobs=3).fit(b[X],b.on/b.off)
    u=(b.on/b.off-m.predict(b[X])).groupby(b.patient_id); us=u.sum(); un=u.size()
    for d in (tr,te):
        f=m.predict(d[X]); ui=(d.patient_id.map(us)/(d.patient_id.map(un)+k)).fillna(0).values
        d['off_eq']=d.on.values/np.clip(f+ui,0.2,None); d['has_u']=d.patient_id.map(un).notna().values
def seg(d):
    off=d.off.notna().values; tre=d.ledd.notna().values; first=(d.groupby('patient_id').cumcount()==0).values; on=d.on.notna().values
    s=np.full(len(d),-1)
    s[off&~tre&first]=0; s[off&~tre&~first]=1; s[off&tre]=2
    s[~off&on&d.has_u.values]=3; s[~off&on&~d.has_u.values]=4
    return s,np.where(off,d.off.values,d.off_eq.values)
def fit_dens(d,bw=0.025):
    s,m=seg(d); q=m/d.target.clip(lower=1).values; ok=d.target.values>8; L=[]
    for k in range(5):
        x=q[(s==k)&ok]; L.append(np.log(gaussian_kde(x,bw_method=bw/x.std())(GRID)+1e-4))
    return np.array(L)
post3.seg=seg  # feats() uses post3.seg
def run(tr,te,folds,pref):
    cols=['p3','p3_sd','p3_b1','p3_b2']
    E=pd.DataFrame(index=tr.index,columns=cols,dtype=float)
    for a,b in folds: E.iloc[b]=post3.feats(tr.iloc[b],fit_dens(tr.iloc[a])).values
    Et=post3.feats(te,fit_dens(tr))
    ren={c:c.replace('p3',pref) for c in cols}
    return E.rename(columns=ren),Et.rename(columns=ren)
if __name__=='__main__':
    tr=pd.read_pickle('tr2.pkl'); te=pd.read_pickle('te2.pkl'); folds=pickle.load(open('folds.pkl','rb')); y=tr.target.values
    add_offeq(tr,te)
    E,Et=run(tr,te,folds,'p4')
    print('p4 OOF rmse %.4f'%np.sqrt(np.mean((E.p4.values-y)**2)),flush=True)
    E['Index']=tr.Index.values; Et['Index']=te.Index.values; E.to_pickle('p4_tr.pkl'); Et.to_pickle('p4_te.pkl')
