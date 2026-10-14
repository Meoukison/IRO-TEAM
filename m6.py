import pickle,numpy as np,pandas as pd,lightgbm as lgb,warnings,sys; warnings.filterwarnings('ignore')
from post import *
Tr=pd.read_pickle('tr7.pkl'); Te=pd.read_pickle('te7.pkl'); folds=pickle.load(open('folds.pkl','rb')); F5=pickle.load(open('F5.pkl','rb'))
G=pd.read_pickle('grid_tr.pkl'); Gt=pd.read_pickle('grid_te.pkl'); GC=[c for c in G if c!='Index']
Tr=Tr.merge(G,on='Index',how='left'); Te=Te.merge(Gt,on='Index',how='left'); y=Tr.target.values
r=lambda p: np.sqrt(np.mean((p-y)**2))
for d in (Tr,Te):
    d['g_all_minus_m']=d.g_all-d.pm_z_c; d['g_off_minus_all']=d.g_off-d.g_all
F6=F5+GC+['g_all_minus_m','g_off_minus_all']
P=dict(objective='regression',learning_rate=0.03,num_leaves=63,min_child_samples=100,colsample_bytree=0.6,subsample=0.8,subsample_freq=1,reg_lambda=10,n_estimators=5000,verbose=-1,random_state=1)
oof=np.zeros(len(Tr)); its=[]
for a,b in folds:
    m=lgb.LGBMRegressor(**P).fit(Tr.iloc[a][F6],y[a],eval_X=Tr.iloc[b][F6],eval_y=y[b],callbacks=[lgb.early_stopping(200,verbose=False)])
    oof[b]=m.predict(Tr.iloc[b][F6]); its.append(m.best_iteration_)
Tr['m6']=oof; Te['m6']=lgb.LGBMRegressor(**{**P,'n_estimators':int(np.mean(its)*1.1)}).fit(Tr[F6],y).predict(Te[F6])
for d in (Tr,Te): d['m6q']=polyproj(d,'m6',2,0)
print('m6 OOF %.4f q %.4f'%(r(oof),r(Tr.m6q.values)),its)
Tr.to_pickle('tr8.pkl'); Te.to_pickle('te8.pkl'); pickle.dump(F6,open('F6.pkl','wb'))
