import sys,pickle,numpy as np,pandas as pd,lightgbm as lgb
sys.path.insert(0,'.')
from day1_model_optimized import add_features
from smooth import curve
tr=pd.read_pickle('tr2.pkl'); te=pd.read_pickle('te2.pkl'); folds=pickle.load(open('folds.pkl','rb'))
def extra(d):
    d=add_features(d)  # sorts by patient, age -> same order as ours
    zc=d.z_off.where(d.z_off.notna(), d.z_on)
    d['z_c']=zc; d['z_c_w']=np.where(d.z_off.notna(),1.0,0.4)
    g=d.groupby('patient_id',sort=False)
    for c in ['z_off','z_on','z_c','s1','gls']:
        d[f'pm_{c}']=g[c].transform('mean'); d[f'pmed_{c}']=g[c].transform('median'); d[f'pstd_{c}']=g[c].transform('std')
    for c,s2 in [('z_c',70),('s1',30)]:
        for deg,pv in [(0,(1e4,)),(1,(1e4,1.0)),(2,(1e4,1.0,0.02))]:
            d[f'cv_{c}_{deg}'],sl=curve(d.assign(**{c:d[c]}).dropna(subset=[c]) if False else d,c,s2,pv,deg=deg) if d[c].notna().all() else (np.nan,np.nan)
            if deg==1: d[f'slope_{c}']=sl
    # weighted fit on z_c with nan handling
    out=np.full(len(d),np.nan)
    for idx in g.indices.values():
        t=d.age.values[idx]; z=d.z_c.values[idx]; w=d.z_c_w.values[idx]; m=~np.isnan(z)
        if m.sum()>=2:
            A=np.vander(t-t.mean(),2,increasing=True); W=w[m]
            b=np.linalg.solve(A[m].T@(A[m]*W[:,None])+np.diag([1e-4,20]),A[m].T@(W*z[m])+np.array([0,20*2.5]))
            out[idx]=A@b
        elif m.sum()==1: out[idx]=z[m][0]
    d['zfit_lin']=out
    return d
Tr=extra(tr.copy()); Te=extra(te.copy())
assert (Tr.Index.values==tr.Index.values).all()
drop={'Index','patient_id','target','cohort','gene','rater_id'}
F=[c for c in Tr.columns if c not in drop and c in Te.columns and pd.api.types.is_numeric_dtype(Tr[c])]
y=Tr.target.values
P=dict(objective='regression',learning_rate=0.03,num_leaves=63,min_child_samples=100,colsample_bytree=0.6,subsample=0.8,subsample_freq=1,reg_lambda=10,n_estimators=4000,verbose=-1)
oof=np.zeros(len(Tr)); its=[]
for a,b in folds:
    m=lgb.LGBMRegressor(**P).fit(Tr.iloc[a][F],y[a],eval_X=Tr.iloc[b][F],eval_y=y[b],callbacks=[lgb.early_stopping(150,verbose=False)])
    oof[b]=m.predict(Tr.iloc[b][F]); its.append(m.best_iteration_)
print('m2 OOF RMSE %.4f'%np.sqrt(np.mean((oof-y)**2)), its, flush=True)
imp=pd.Series(m.feature_importances_,F).sort_values(ascending=False); print(imp.head(20).to_dict())
mt=lgb.LGBMRegressor(**{**P,'n_estimators':int(np.mean(its)*1.1)}).fit(Tr[F],y)
Tr['m2']=oof; Te['m2']=mt.predict(Te[F])
Tr.to_pickle('tr3.pkl'); Te.to_pickle('te3.pkl'); pickle.dump(F,open('F3.pkl','wb'))
