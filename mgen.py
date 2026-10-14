import pickle,numpy as np,pandas as pd,lightgbm as lgb,warnings; warnings.filterwarnings('ignore')
from post import *
Tr=pd.read_pickle('tr10.pkl'); Te=pd.read_pickle('te10.pkl'); folds=pickle.load(open('folds.pkl','rb')); F7=pickle.load(open('F7.pkl','rb'))
G=pd.read_pickle('p4_tr.pkl').merge(pd.read_pickle('p5_tr.pkl'),on='Index'); Gt=pd.read_pickle('p4_te.pkl').merge(pd.read_pickle('p5_te.pkl'),on='Index')
for d,g in ((Tr,G),(Te,Gt)):
    g['p5_pm']=d.set_index('Index').loc[g.Index].patient_id.values; g['p5_pm']=g.groupby('p5_pm').p5.transform('mean')
    g['p5_minus_p3']=g.p5.values-d.set_index('Index').loc[g.Index].p3.values
G[[c for c in G if c!='Index']].join(G.Index).to_pickle('gen_tr.pkl'); Gt.to_pickle('gen_te.pkl')
GC=[c for c in G if c!='Index']
Tr=Tr.merge(G,on='Index',how='left'); Te=Te.merge(Gt,on='Index',how='left'); y=Tr.target.values
r=lambda p: np.sqrt(np.mean((p-y)**2))
FG=F7+GC
P=dict(objective='regression',learning_rate=0.03,num_leaves=63,min_child_samples=100,colsample_bytree=0.6,subsample=0.8,subsample_freq=1,reg_lambda=10,n_estimators=5000,verbose=-1,random_state=1,n_jobs=3)
oof=np.zeros(len(Tr)); its=[]
for a,b in folds:
    m=lgb.LGBMRegressor(**P).fit(Tr.iloc[a][FG],y[a],eval_X=Tr.iloc[b][FG],eval_y=y[b],callbacks=[lgb.early_stopping(200,verbose=False)])
    oof[b]=m.predict(Tr.iloc[b][FG]); its.append(m.best_iteration_)
Tr['mgen']=oof; Te['mgen']=lgb.LGBMRegressor(**{**P,'n_estimators':int(np.mean(its)*1.1)}).fit(Tr[FG],y).predict(Te[FG])
for d in (Tr,Te): d['mgen_q']=polyproj(d,'mgen',2,0)
print('mgen OOF %.4f q %.4f'%(r(oof),r(Tr.mgen_q.values)),its,flush=True)
if r(Tr.mgen_q.values)<3.1154:
    Tr.to_pickle('tr_gen.pkl'); Te.to_pickle('te_gen.pkl'); pickle.dump(FG,open('FG.pkl','wb')); save(Te,'mgen_q','41_generative')
