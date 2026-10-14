import pickle,numpy as np,pandas as pd,lightgbm as lgb,warnings; warnings.filterwarnings('ignore')
from post import *
Tr=pd.read_pickle('tr_pair2.pkl'); Te=pd.read_pickle('te_pair2.pkl'); folds=pickle.load(open('folds.pkl','rb')); F=pickle.load(open('Fpair2.pkl','rb'))
G=pd.read_pickle('pair3_tr.pkl').merge(pd.read_pickle('p6_tr.pkl'),on='Index'); Gt=pd.read_pickle('pair3_te.pkl').merge(pd.read_pickle('p6_te.pkl'),on='Index')
pc=[c for c in pd.read_pickle('pair3_tr.pkl') if c!='Index']
ren={c:'v3_'+c for c in pc}; G=G.rename(columns=ren); Gt=Gt.rename(columns=ren)
Tr=Tr.merge(G,on='Index',how='left'); Te=Te.merge(Gt,on='Index',how='left'); y=Tr.target.values
for d in (Tr,Te):
    for c in ['pr_mean','pr_med','pr_ivw']: d[f'v3_{c}_abs']=d['v3_'+c]+d.p6
FF=F+list(ren.values())+['p6','p6_sd','p6_b1']+[f'v3_{c}_abs' for c in ['pr_mean','pr_med','pr_ivw']]
r=lambda p: np.sqrt(np.mean((p-y)**2))
P=dict(objective='regression',learning_rate=0.03,num_leaves=63,min_child_samples=100,colsample_bytree=0.6,subsample=0.8,subsample_freq=1,reg_lambda=10,n_estimators=10000,verbose=-1,random_state=1,n_jobs=4)
oof=np.zeros(len(Tr)); its=[]
for a,b in folds:
    m=lgb.LGBMRegressor(**P).fit(Tr.iloc[a][FF],y[a],eval_X=Tr.iloc[b][FF],eval_y=y[b],callbacks=[lgb.early_stopping(200,verbose=False)])
    oof[b]=m.predict(Tr.iloc[b][FF]); its.append(m.best_iteration_); print('fold done',m.best_iteration_,flush=True)
Tr['mpair3']=oof; Te['mpair3']=lgb.LGBMRegressor(**{**P,'n_estimators':int(np.mean(its)*1.1)}).fit(Tr[FF],y).predict(Te[FF])
for d in (Tr,Te): d['mpair3_q']=polyproj(d,'mpair3',2,0)
print('mpair3 OOF %.4f q %.4f'%(r(oof),r(Tr.mpair3_q.values)),its)
if r(Tr.mpair3_q.values)<2.8414:
    Tr.to_pickle('tr_pair3.pkl'); Te.to_pickle('te_pair3.pkl'); pickle.dump(FF,open('Fpair3.pkl','wb')); save(Te,'mpair3_q','47_pairwise_v3')
