import pickle,numpy as np,pandas as pd,lightgbm as lgb,warnings; warnings.filterwarnings('ignore')
from post import *
Tr=pd.read_pickle('tr_gen.pkl'); Te=pd.read_pickle('te_gen.pkl'); folds=pickle.load(open('folds.pkl','rb')); FG=pickle.load(open('FG.pkl','rb'))
G=pd.read_pickle('p6_tr.pkl'); Gt=pd.read_pickle('p6_te.pkl')
Tr=Tr.merge(G,on='Index',how='left'); Te=Te.merge(Gt,on='Index',how='left'); y=Tr.target.values
for d in (Tr,Te): d['p6_pm']=d.groupby('patient_id').p6.transform('mean'); d['p6_minus_p5']=d.p6-d.p5
FG2=FG+['p6','p6_sd','p6_b1','p6_pm','p6_minus_p5']
r=lambda p: np.sqrt(np.mean((p-y)**2))
P=dict(objective='regression',learning_rate=0.03,num_leaves=63,min_child_samples=100,colsample_bytree=0.6,subsample=0.8,subsample_freq=1,reg_lambda=10,n_estimators=5000,verbose=-1,random_state=1,n_jobs=3)
oof=np.zeros(len(Tr)); its=[]
for a,b in folds:
    m=lgb.LGBMRegressor(**P).fit(Tr.iloc[a][FG2],y[a],eval_X=Tr.iloc[b][FG2],eval_y=y[b],callbacks=[lgb.early_stopping(200,verbose=False)])
    oof[b]=m.predict(Tr.iloc[b][FG2]); its.append(m.best_iteration_)
Tr['mgen2']=oof; Te['mgen2']=lgb.LGBMRegressor(**{**P,'n_estimators':int(np.mean(its)*1.1)}).fit(Tr[FG2],y).predict(Te[FG2])
for d in (Tr,Te): d['mgen2_q']=polyproj(d,'mgen2',2,0)
print('mgen2 OOF %.4f q %.4f'%(r(oof),r(Tr.mgen2_q.values)),its,flush=True)
if r(Tr.mgen2_q.values)<2.8701:
    Tr['mgen_q']=Tr.mgen2_q; Te['mgen_q']=Te.mgen2_q
    Tr.to_pickle('tr_gen.pkl'); Te.to_pickle('te_gen.pkl'); pickle.dump(FG2,open('FG.pkl','wb')); save(Te,'mgen_q','41_generative')
