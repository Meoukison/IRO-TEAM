import pandas as pd,numpy as np,pickle,lightgbm as lgb
import post3,post4
from post3 import G,logf
from post4 import add_offeq,seg,fit_dens
def pcov(d):
    g=d.groupby('patient_id',sort=False)
    return pd.DataFrame({'am':g.age.mean(),'span':g.age.max()-g.age.min(),'adx':g.age_at_diagnosis.first(),'sex':g.sexM.first(),
      'gene':g.gene.first().map({'No Mutation':0,'LRRK2+':1,'GBA+':2,'OTHER+':3}).fillna(-1),'coh':(g.cohort.first()=='B').astype(int),
      'lm':g.ledd.mean(),'lmax':g.ledd.max(),'lf':g.ledd.count()/g.size(),'n':g.size(),'offf':g.off.count()/g.size(),'tsf':g.time_since_intake_off.count()/g.size(),
      'l0':d[d.ledd.notna()].groupby('patient_id').age.min()})
def curve_coefs(d):
    C={}
    for pid,idx in d.groupby('patient_id',sort=False).indices.items():
        t=d.age.values[idx]; C[pid]=np.polyfit(t-t.mean(),d.target.values[idx],2)[::-1]
    return pd.DataFrame(C,index=['b0','b1','b2']).T
def prior(dfit,dpred):
    Pf=pcov(dfit); Pp=pcov(dpred); C=curve_coefs(dfit).loc[Pf.index]; out={}
    for c in ['b0','b1']:
        m=lgb.LGBMRegressor(n_estimators=400,learning_rate=0.03,num_leaves=15,min_child_samples=50,verbose=-1,n_jobs=3).fit(Pf,C[c])
        res=C[c]-m.predict(Pf); out[c]=pd.Series(m.predict(Pp),Pp.index); out[c+'_sd']=res.std()*1.15  # inflate for in-sample optimism
    return out
def feats(d,L,pr,use_b0=True):
    s,m=seg(d); n=len(d); out={k:np.full(n,np.nan) for k in ['p5','p5_sd','p5_b1']}
    for pid,idx in d.groupby('patient_id',sort=False).indices.items():
        k=s[idx]>=0; t=d.age.values[idx]; tm=t.mean(); tc=t-tm; lo,hi=tc.min(),tc.max()
        ok=(G[1]+2*G[2]*lo>=0)&(G[1]+2*G[2]*hi>=0); g0,g1,g2=G[0][ok],G[1][ok],G[2][ok]
        ll=-0.5*(g1-pr['b1'][pid])**2/pr['b1_sd']**2-0.5*g2**2/0.15**2
        if use_b0: ll+=-0.5*(g0-pr['b0'][pid])**2/pr['b0_sd']**2
        T=np.clip(g0[:,None]+g1[:,None]*tc[None,:]+g2[:,None]*tc[None,:]**2,0.5,None)
        for j in np.where(k)[0]: ll+=logf(L,s[idx][j],m[idx][j]/T[:,j])-np.log(T[:,j])
        w=np.exp(ll-ll.max()); w/=w.sum(); mu=w@T
        out['p5'][idx]=mu; out['p5_sd'][idx]=np.sqrt(np.clip(w@T**2-mu**2,0,None)); out['p5_b1'][idx]=w@g1
    return pd.DataFrame(out,index=d.index)
if __name__=='__main__':
    tr=pd.read_pickle('tr2.pkl'); te=pd.read_pickle('te2.pkl'); folds=pickle.load(open('folds.pkl','rb')); y=tr.target.values
    add_offeq(tr,te)
    E=pd.DataFrame(index=tr.index,columns=['p5','p5_sd','p5_b1'],dtype=float)
    for a,b in folds:
        A=tr.iloc[a]; pr=prior(A,tr.iloc[b]); print('prior sd b0 %.2f b1 %.2f'%(pr['b0_sd'],pr['b1_sd']),flush=True)
        E.iloc[b]=feats(tr.iloc[b],fit_dens(A),pr).values
    Et=feats(te,fit_dens(tr),prior(tr,te))
    print('p5 OOF rmse %.4f'%np.sqrt(np.mean((E.p5.values-y)**2)),flush=True)
    E['Index']=tr.Index.values; Et['Index']=te.Index.values; E.to_pickle('p5_tr.pkl'); Et.to_pickle('p5_te.pkl')
