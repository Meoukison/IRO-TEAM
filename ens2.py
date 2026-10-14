import pickle,numpy as np,pandas as pd,lightgbm as lgb,warnings,os; warnings.filterwarnings('ignore')
from catboost import CatBoostRegressor
from post import *
Tr=pd.read_pickle('tr_gen.pkl'); Te=pd.read_pickle('te_gen.pkl'); folds=pickle.load(open('folds.pkl','rb')); F=pickle.load(open('FG.pkl','rb')); y=Tr.target.values
r=lambda p: np.sqrt(np.mean((p-y)**2))
store='ens2_preds.pkl'; S=pickle.load(open(store,'rb')) if os.path.exists(store) else {}
S['mgen']=(Tr.mgen.values,Te.mgen.values)
base_tr=Tr.p5.values; base_te=Te.p5.values
cfgs=[
 ('lgb15',dict(objective='regression',learning_rate=0.03,num_leaves=15,min_child_samples=200,colsample_bytree=0.4,subsample=0.7,subsample_freq=1,reg_lambda=30,n_estimators=8000,verbose=-1,random_state=12,n_jobs=3),'raw'),
 ('lgb_res',dict(objective='regression',learning_rate=0.03,num_leaves=63,min_child_samples=100,colsample_bytree=0.6,subsample=0.8,subsample_freq=1,reg_lambda=10,n_estimators=5000,verbose=-1,random_state=11,n_jobs=3),'res'),
 ('cat8',None,'raw'),
]
def tgt(kind,idx=None):
    b=base_tr if idx is None else base_tr[idx]; yy=y if idx is None else y[idx]
    return {'raw':yy,'res':yy-b,'ratio':yy/np.clip(b,3,None)}[kind]
def inv(kind,p,b): return {'raw':p,'res':p+b,'ratio':p*np.clip(b,3,None)}[kind]
for name,P,kind in cfgs:
    if name in S: print(name,'cached',flush=True); continue
    oof=np.zeros(len(Tr)); its=[]
    for a,b in folds:
        if P is None:
            m=CatBoostRegressor(iterations=6000,learning_rate=0.05,depth=8,l2_leaf_reg=8,verbose=0,thread_count=3,random_seed=4,od_type='Iter',od_wait=200)
            m.fit(Tr.iloc[a][F],tgt(kind,a),eval_set=(Tr.iloc[b][F],tgt(kind,b)),use_best_model=True); its.append(m.get_best_iteration())
        else:
            m=lgb.LGBMRegressor(**P).fit(Tr.iloc[a][F],tgt(kind,a),eval_X=Tr.iloc[b][F],eval_y=tgt(kind,b),callbacks=[lgb.early_stopping(200,verbose=False)]); its.append(m.best_iteration_)
        oof[b]=inv(kind,m.predict(Tr.iloc[b][F]),base_tr[b])
    n=int(np.mean(its)*1.1)
    mt=CatBoostRegressor(iterations=n,learning_rate=0.05,depth=8,l2_leaf_reg=8,verbose=0,thread_count=3,random_seed=4) if P is None else lgb.LGBMRegressor(**{**P,'n_estimators':n})
    mt.fit(Tr[F],tgt(kind)); pte=inv(kind,mt.predict(Te[F]),base_te)
    S[name]=(oof,pte); pickle.dump(S,open(store,'wb'))
    print(name,'OOF %.4f'%r(oof),its,flush=True)
