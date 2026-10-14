import pickle,numpy as np,pandas as pd,lightgbm as lgb,warnings,sys; warnings.filterwarnings('ignore')
from post import *
Tr=pd.read_pickle('tr8.pkl'); Te=pd.read_pickle('te8.pkl'); folds=pickle.load(open('folds.pkl','rb')); F6=pickle.load(open('F6.pkl','rb'))
G=pd.read_pickle('p3_tr.pkl'); Gt=pd.read_pickle('p3_te.pkl'); GC=[c for c in G if c!='Index']
Tr=Tr.merge(G,on='Index',how='left'); Te=Te.merge(Gt,on='Index',how='left'); y=Tr.target.values
r=lambda p: np.sqrt(np.mean((p-y)**2))
for d in (Tr,Te): d['p3_minus_g']=d.p3-d.g_all; d['p3_pm']=d.groupby('patient_id').p3.transform('mean')
F7=F6+GC+['p3_minus_g','p3_pm']
P=dict(objective='regression',learning_rate=0.03,num_leaves=63,min_child_samples=100,colsample_bytree=0.6,subsample=0.8,subsample_freq=1,reg_lambda=10,n_estimators=5000,verbose=-1,random_state=1,n_jobs=4)
oof=np.zeros(len(Tr)); its=[]
for a,b in folds:
    m=lgb.LGBMRegressor(**P).fit(Tr.iloc[a][F7],y[a],eval_X=Tr.iloc[b][F7],eval_y=y[b],callbacks=[lgb.early_stopping(200,verbose=False)])
    oof[b]=m.predict(Tr.iloc[b][F7]); its.append(m.best_iteration_)
Tr['m7']=oof; Te['m7']=lgb.LGBMRegressor(**{**P,'n_estimators':int(np.mean(its)*1.1)}).fit(Tr[F7],y).predict(Te[F7])
for d in (Tr,Te): d['m7q']=polyproj(d,'m7',2,0)
print('m7 OOF %.4f q %.4f'%(r(oof),r(Tr.m7q.values)),its)
Tr.to_pickle('tr10.pkl'); Te.to_pickle('te10.pkl'); pickle.dump(F7,open('F7.pkl','wb'))
