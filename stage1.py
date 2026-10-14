from load import *
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
tr=load('train'); te=load('test')
def feats(d):
    f=pd.DataFrame(index=d.index)
    for c in ['sexM','age_at_diagnosis','age','dur','ledd','time_since_intake_on','time_since_intake_off','on','off','on_o','off_o']: f[c]=d[c]
    f['gene']=d.gene.fillna('NA').astype('category'); f['cohort']=d.cohort.astype('category'); f['rater']=d.rater_id.astype('category')
    f['ledd_na']=d.ledd.isna().astype(int)
    return f
cats=sorted(set(tr.rater_id)|set(te.rater_id)); genes=sorted(set(tr.gene.fillna('NA'))|set(te.gene.fillna('NA')))
Ftr,Fte=feats(tr),feats(te)
for F in (Ftr,Fte):
    F['rater']=pd.Categorical(F.rater.astype(str),categories=cats); F['gene']=pd.Categorical(F.gene.astype(str),categories=genes); F['cohort']=pd.Categorical(F.cohort.astype(str),categories=['A','B'])
y=tr.target.values
folds=list(GroupKFold(5,shuffle=True,random_state=42).split(Ftr,y,tr.patient_id))
P=dict(objective='regression',learning_rate=0.03,num_leaves=63,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,n_estimators=2000,verbose=-1)
oof=np.zeros(len(tr))
for a,b in folds:
    m=lgb.LGBMRegressor(**P).fit(Ftr.iloc[a],y[a],eval_X=Ftr.iloc[b],eval_y=y[b],callbacks=[lgb.early_stopping(100,verbose=False)])
    oof[b]=m.predict(Ftr.iloc[b])
print('stage1 single-visit OOF RMSE %.4f'%np.sqrt(np.mean((oof-y)**2)))
m=lgb.LGBMRegressor(**{**P,'n_estimators':int(m.best_iteration_*1.1)}).fit(Ftr,y)
tr['s1']=oof; te['s1']=m.predict(Fte)
tr.to_pickle('tr1.pkl'); te.to_pickle('te1.pkl'); import pickle; pickle.dump(folds,open('folds.pkl','wb'))
