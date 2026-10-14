import pickle,numpy as np,pandas as pd,lightgbm as lgb,warnings; warnings.filterwarnings('ignore')
tr=pd.read_pickle('tr2.pkl'); te=pd.read_pickle('te2.pkl'); folds=pickle.load(open('folds.pkl','rb'))
for d,f in ((tr,'p3_tr.pkl'),(te,'p3_te.pkl')):
    P=pd.read_pickle(f); assert (P.Index.values==d.Index.values).all()
    for c in ['p3','p3_sd','p3_b1','p3_b2']: d[c]=P[c].values
    Q5=pd.read_pickle(f.replace('p3','gen')); assert (Q5.Index.values==d.Index.values).all()
    for c in ['p5','p5_sd','p5_b1']: d[c]=Q5[c].values
def base(d):
    g=d.groupby('patient_id',sort=False)
    d['first']=(g.cumcount()==0).astype(int); d['nvis']=g.age.transform('size'); d['vis']=g.cumcount()
    d['treated']=d.ledd.notna().astype(int)
    return d
from post4 import add_offeq
add_offeq(tr,te)
tr=base(tr); te=base(te)
SRC=['age','off','on','off_eq','ledd','time_since_intake_on','time_since_intake_off','z_off','z_on','first','treated','vis','p3','p5']
TGT=['age','dur','ledd','nvis','vis','p3','p3_sd','p3_b1','p3_b2','p5','p5_sd','p5_b1','age_at_diagnosis','sexM','treated']
def pairs(d):
    I=[];J=[]
    for idx in d.groupby('patient_id',sort=False).indices.values():
        a,b=np.meshgrid(idx,idx,indexing='ij'); I.append(a.ravel()); J.append(b.ravel())
    I=np.concatenate(I); J=np.concatenate(J)
    X=pd.DataFrame({**{'t_'+c:d[c].values[I] for c in TGT},**{'s_'+c:d[c].values[J] for c in SRC}})
    X['dt']=X.t_age-X.s_age; X['self']=(I==J).astype(int)
    X['s_off_rel']=X.s_off/X.s_p3.clip(lower=1); X['s_on_rel']=X.s_on/X.s_p3.clip(lower=1)
    X['s_off_proj']=X.s_off*X.t_p5/X.s_p5.clip(lower=1); X['s_on_proj']=X.s_on*X.t_p5/X.s_p5.clip(lower=1); X['s_offeq_proj']=X.s_off_eq*X.t_p5/X.s_p5.clip(lower=1); X['s_offeq_rel']=X.s_off_eq/X.s_p5.clip(lower=1)
    return X,I
P=dict(objective='regression',learning_rate=0.05,num_leaves=63,min_child_samples=200,colsample_bytree=0.8,subsample=0.5,subsample_freq=1,reg_lambda=10,n_estimators=600,verbose=-1,n_jobs=3)
Q=dict(P,objective='quantile',alpha=0.85)
def agg(X,I,pred,predq,n):
    df=pd.DataFrame({'i':I,'p':pred,'q':predq,'has':X.s_off.notna().values|X.s_on.notna().values,'adt':np.abs(X.dt.values)})
    df=df[df.has]; g=df.groupby('i')
    w=1/(1+df.adt); df['wp']=df.p*w; df['w']=w; gw=df.groupby('i')
    out=pd.DataFrame(index=range(n))
    out['pr_mean']=g.p.mean(); out['pr_med']=g.p.median(); out['pr_max']=g.p.max(); out['pr_min']=g.p.min(); out['pr_std']=g.p.std()
    out['pr_wmean']=gw.wp.sum()/gw.w.sum(); out['pq_mean']=g.q.mean(); out['pq_max']=g.q.max()
    return out
def run(trd,ted):
    Xa,Ia=pairs(trd); ya=trd.target.values[Ia]-trd.p5.values[Ia]
    Xb,Ib=pairs(ted)
    m=lgb.LGBMRegressor(**P).fit(Xa,ya); mq=lgb.LGBMRegressor(**Q).fit(Xa,ya)
    return agg(Xb,Ib,m.predict(Xb),mq.predict(Xb),len(ted))
cols=None; OUT=[]
oof=None
for k,(a,b) in enumerate(folds):
    o=run(tr.iloc[a].reset_index(drop=True),tr.iloc[b].reset_index(drop=True)); o.index=b
    oof=o if oof is None else pd.concat([oof,o]); print('fold',k,flush=True)
oof=oof.sort_index()
y=tr.target.values; print('pr_mean+p5 rmse %.4f, p5 %.4f'%(np.sqrt(np.nanmean((oof.pr_mean+tr.p5-y)**2)),np.sqrt(np.mean((tr.p5-y)**2))),flush=True)
tst=run(tr.reset_index(drop=True),te.reset_index(drop=True))
oof['Index']=tr.Index.values; tst['Index']=te.Index.values
oof.to_pickle('pair2_tr.pkl'); tst.to_pickle('pair2_te.pkl')
