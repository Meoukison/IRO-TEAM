import pickle,numpy as np,pandas as pd,lightgbm as lgb,warnings; warnings.filterwarnings('ignore')
from post import *
Tr=pd.read_pickle('tr_gen.pkl'); Te=pd.read_pickle('te_gen.pkl'); folds=pickle.load(open('folds.pkl','rb')); FG=pickle.load(open('FG.pkl','rb'))
G=pd.read_pickle('pair2_tr.pkl'); Gt=pd.read_pickle('pair2_te.pkl'); GC=[c for c in G if c!='Index']
G=G.rename(columns={c:'v2_'+c for c in GC}); Gt=Gt.rename(columns={c:'v2_'+c for c in GC}); GC=['v2_'+c for c in GC]
Tr=Tr.merge(G,on='Index',how='left'); Te=Te.merge(Gt,on='Index',how='left'); y=Tr.target.values
for d in (Tr,Te):
    for c in ['pr_mean','pr_med','pr_wmean','pq_mean']: d['v2_'+c+'_abs']=d['v2_'+c]+d.p5
FF=FG+GC+['v2_'+c+'_abs' for c in ['pr_mean','pr_med','pr_wmean','pq_mean']]
r=lambda p: np.sqrt(np.mean((p-y)**2))
P=dict(objective='regression',learning_rate=0.03,num_leaves=63,min_child_samples=100,colsample_bytree=0.6,subsample=0.8,subsample_freq=1,reg_lambda=10,n_estimators=5000,verbose=-1,random_state=1,n_jobs=3)
oof=np.zeros(len(Tr)); its=[]
for a,b in folds:
    m=lgb.LGBMRegressor(**P).fit(Tr.iloc[a][FF],y[a],eval_X=Tr.iloc[b][FF],eval_y=y[b],callbacks=[lgb.early_stopping(200,verbose=False)])
    oof[b]=m.predict(Tr.iloc[b][FF]); its.append(m.best_iteration_)
Tr['mpair2']=oof; Te['mpair2']=lgb.LGBMRegressor(**{**P,'n_estimators':int(np.mean(its)*1.1)}).fit(Tr[FF],y).predict(Te[FF])
for d in (Tr,Te): d['mpair2_q']=polyproj(d,'mpair2',2,0)
print('mpair2 OOF %.4f q %.4f (mgen_q %.4f)'%(r(oof),r(Tr.mpair2_q.values),r(Tr.mgen_q.values)),its)
if r(Tr.mpair2_q.values)<2.8701:
    Tr.to_pickle('tr_pair2.pkl'); Te.to_pickle('te_pair2.pkl'); pickle.dump(FF,open('Fpair2.pkl','wb')); save(Te,'mpair2_q','44_pairwise_gen')
