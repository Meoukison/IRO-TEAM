import pickle,numpy as np,pandas as pd,lightgbm as lgb,warnings; warnings.filterwarnings('ignore')
from post import *
Tr=pd.read_pickle('tr3.pkl'); Te=pd.read_pickle('te3.pkl'); folds=pickle.load(open('folds.pkl','rb')); F=pickle.load(open('F3.pkl','rb')); y=Tr.target.values
r=lambda p: np.sqrt(np.mean((p-y)**2))
E=pd.read_pickle('env_tr.pkl').merge(pd.read_pickle('mle_tr.pkl'),on='Index'); Et=pd.read_pickle('env_te.pkl').merge(pd.read_pickle('mle_te.pkl'),on='Index')
EC=[c for c in E if c!='Index']
Tr=Tr.drop(columns=[c for c in EC if c in Tr]).merge(E,on='Index',how='left'); Te=Te.drop(columns=[c for c in EC if c in Te]).merge(Et,on='Index',how='left')
for d in (Tr,Te):
    g=d.groupby('patient_id',sort=False)
    for c in ['mle_off','mle_lin','env_q95']: d['pm_'+c]=g[c].transform('mean')
    d['mle_minus_zc']=d.mle_lin-d.pm_z_c
EC+=['pm_mle_off','pm_mle_lin','pm_env_q95','mle_minus_zc']
F5=F+EC
P=dict(objective='regression',learning_rate=0.03,num_leaves=63,min_child_samples=100,colsample_bytree=0.6,subsample=0.8,subsample_freq=1,reg_lambda=10,n_estimators=5000,verbose=-1,random_state=1)
oof=np.zeros(len(Tr)); its=[]
for a,b in folds:
    m=lgb.LGBMRegressor(**P).fit(Tr.iloc[a][F5],y[a],eval_X=Tr.iloc[b][F5],eval_y=y[b],callbacks=[lgb.early_stopping(200,verbose=False)])
    oof[b]=m.predict(Tr.iloc[b][F5]); its.append(m.best_iteration_)
Tr['m5']=oof; Te['m5']=lgb.LGBMRegressor(**{**P,'n_estimators':int(np.mean(its)*1.1)}).fit(Tr[F5],y).predict(Te[F5])
for d in (Tr,Te): d['m5q']=polyproj(d,'m5',2,0)
print('m5 OOF %.4f q %.4f'%(r(oof),r(Tr.m5q.values)),its)
imp=pd.Series(m.feature_importances_,F5); print(imp[EC].sort_values(ascending=False).to_dict())
Tr.to_pickle('tr7.pkl'); Te.to_pickle('te7.pkl'); pickle.dump(F5,open('F5.pkl','wb'))
