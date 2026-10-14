import pickle,numpy as np,pandas as pd,lightgbm as lgb,warnings,os; warnings.filterwarnings('ignore')
from catboost import CatBoostRegressor
Tr=pd.read_pickle('tr_pair2.pkl'); Te=pd.read_pickle('te_pair2.pkl'); G=pd.read_pickle('tr_gen.pkl'); Gt=pd.read_pickle('te_gen.pkl')
P6=['p6','p6_sd','p6_b1','p6_pm','p6_minus_p5']
Tr=Tr.drop(columns=[c for c in P6+['mgen'] if c in Tr]).merge(G[['Index','mgen']+P6],on='Index'); Te=Te.drop(columns=[c for c in P6+['mgen'] if c in Te]).merge(Gt[['Index','mgen']+P6],on='Index')
folds=pickle.load(open('folds.pkl','rb')); F=pickle.load(open('Fpair2.pkl','rb'))+P6; y=Tr.target.values
assert (Tr.Index.values==G.Index.values).all()
r=lambda p: np.sqrt(np.mean((p-y)**2))
store='ens3_preds.pkl'; S=pickle.load(open(store,'rb')) if os.path.exists(store) else {}
S['mpair2']=(Tr.mpair2.values,Te.mpair2.values); S['mgen']=(Tr.mgen.values,Te.mgen.values)
E2=pickle.load(open('ens2_preds.pkl','rb'))
for k in ('lgb15','lgb_res'): S['e2_'+k]=E2[k]
pickle.dump(S,open(store,'wb'))
base_tr=Tr.p6.values; base_te=Te.p6.values
cfgs=[
 ('n_res6',dict(objective='regression',learning_rate=0.03,num_leaves=63,min_child_samples=100,colsample_bytree=0.6,subsample=0.8,subsample_freq=1,reg_lambda=10,n_estimators=5000,verbose=-1,random_state=21,n_jobs=3),'res'),
 ('n_lgb15',dict(objective='regression',learning_rate=0.03,num_leaves=15,min_child_samples=200,colsample_bytree=0.4,subsample=0.7,subsample_freq=1,reg_lambda=30,n_estimators=8000,verbose=-1,random_state=22,n_jobs=3),'raw'),
 ('n_cat8',None,'raw'),
]
def tgt(kind,idx=None):
    b=base_tr if idx is None else base_tr[idx]; yy=y if idx is None else y[idx]
    return yy-b if kind=='res' else yy
def inv(kind,p,b): return p+b if kind=='res' else p
for name,P,kind in cfgs:
    if name in S: continue
    oof=np.zeros(len(Tr)); its=[]
    for a,b in folds:
        if P is None:
            m=CatBoostRegressor(iterations=3000,learning_rate=0.08,depth=8,l2_leaf_reg=8,verbose=0,thread_count=3,random_seed=4,od_type='Iter',od_wait=150)
            m.fit(Tr.iloc[a][F],tgt(kind,a),eval_set=(Tr.iloc[b][F],tgt(kind,b)),use_best_model=True); its.append(m.get_best_iteration())
        else:
            m=lgb.LGBMRegressor(**P).fit(Tr.iloc[a][F],tgt(kind,a),eval_X=Tr.iloc[b][F],eval_y=tgt(kind,b),callbacks=[lgb.early_stopping(200,verbose=False)]); its.append(m.best_iteration_)
        oof[b]=inv(kind,m.predict(Tr.iloc[b][F]),base_tr[b])
    n=int(np.mean(its)*1.1)
    mt=CatBoostRegressor(iterations=n,learning_rate=0.08,depth=8,l2_leaf_reg=8,verbose=0,thread_count=3,random_seed=4) if P is None else lgb.LGBMRegressor(**{**P,'n_estimators':n})
    mt.fit(Tr[F],tgt(kind)); S[name]=(oof,inv(kind,mt.predict(Te[F]),base_te)); pickle.dump(S,open(store,'wb'))
    print(name,'OOF %.4f'%r(oof),its,flush=True)
