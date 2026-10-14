import pickle,numpy as np,pandas as pd,lightgbm as lgb,warnings; warnings.filterwarnings('ignore')
from post4 import add_offeq, seg as seg4
tr=pd.read_pickle('tr2.pkl'); te=pd.read_pickle('te2.pkl'); folds=pickle.load(open('folds.pkl','rb'))
for d,s in ((tr,'tr'),(te,'te')):
    P=pd.read_pickle(f'p6_{s}.pkl'); assert (P.Index.values==d.Index.values).all()
    for c in ['p6','p6_sd','p6_b1']: d[c]=P[c].values
add_offeq(tr,te)
for d in (tr,te):
    g=d.groupby('patient_id',sort=False)
    d['first']=(g.cumcount()==0).astype(int); d['nvis']=g.age.transform('size'); d['vis']=g.cumcount(); d['treated']=d.ledd.notna().astype(int)
    d['seg']=seg4(d)[0]
SRC=['age','off','on','off_eq','ledd','time_since_intake_on','time_since_intake_off','z_off','z_on','first','treated','vis','p6','p6_sd','seg']
TGT=['age','dur','ledd','nvis','vis','p6','p6_sd','p6_b1','age_at_diagnosis','sexM','treated','first']
def pairs(d):
    I=[];J=[]
    for idx in d.groupby('patient_id',sort=False).indices.values():
        a,b=np.meshgrid(idx,idx,indexing='ij'); I.append(a.ravel()); J.append(b.ravel())
    I=np.concatenate(I); J=np.concatenate(J)
    X=pd.DataFrame({**{'t_'+c:d[c].values[I] for c in TGT},**{'s_'+c:d[c].values[J] for c in SRC}})
    X['dt']=X.t_age-X.s_age; X['adt']=X.dt.abs(); X['dt2']=X.dt**2; X['self']=(I==J).astype(int)
    sp=X.s_p6.clip(lower=1)
    for c in ['off','on','off_eq']:
        X[f'rel_{c}']=X['s_'+c]/sp; X[f'impl_{c}']=X.t_p6*X[f'rel_{c}']; X[f'implres_{c}']=X[f'impl_{c}']-X.t_p6
    return X,I
P=dict(objective='regression',learning_rate=0.05,num_leaves=63,min_child_samples=200,colsample_bytree=0.8,subsample=0.5,subsample_freq=1,reg_lambda=10,n_estimators=1500,verbose=-1,n_jobs=4)
R=dict(objective='regression',learning_rate=0.05,num_leaves=31,min_child_samples=500,colsample_bytree=0.8,subsample=0.5,subsample_freq=1,n_estimators=300,verbose=-1,n_jobs=4)
def run(trd,ted):
    Xa,Ia=pairs(trd); ya=trd.target.values[Ia]-trd.p6.values[Ia]
    Xb,Ib=pairs(ted)
    m=lgb.LGBMRegressor(**P).fit(Xa,ya)
    ra=np.abs(ya-m.predict(Xa)); mr=lgb.LGBMRegressor(**R).fit(Xa,ra)
    p=m.predict(Xb); rel=np.clip(mr.predict(Xb),0.3,None)
    df=pd.DataFrame({'i':Ib,'p':p,'rel':rel,'has':Xb.s_off.notna().values|Xb.s_on.notna().values}); df=df[df.has]
    df['w']=1/df.rel**2; df['wp']=df.p*df.w; g=df.groupby('i')
    out=pd.DataFrame(index=range(len(ted)))
    out['pr_mean']=g.p.mean(); out['pr_med']=g.p.median(); out['pr_std']=g.p.std(); out['pr_max']=g.p.max(); out['pr_min']=g.p.min()
    out['pr_ivw']=g.wp.sum()/g.w.sum(); out['pr_ivw_sd']=1/np.sqrt(g.w.sum()); out['rel_min']=g.rel.min(); out['rel_mean']=g.rel.mean()
    return out
oof=[]
for k,(a,b) in enumerate(folds):
    o=run(tr.iloc[a].reset_index(drop=True),tr.iloc[b].reset_index(drop=True)); o.index=b; oof.append(o); print('fold',k,flush=True)
oof=pd.concat(oof).sort_index(); y=tr.target.values
for c in ['pr_mean','pr_ivw']: print(c,'+p6 rmse %.4f'%np.sqrt(np.mean((oof[c]+tr.p6-y)**2)))
print('p6 %.4f'%np.sqrt(np.mean((tr.p6-y)**2)),flush=True)
tst=run(tr.reset_index(drop=True),te.reset_index(drop=True))
oof['Index']=tr.Index.values; tst['Index']=te.Index.values; oof.to_pickle('pair3_tr.pkl'); tst.to_pickle('pair3_te.pkl')
