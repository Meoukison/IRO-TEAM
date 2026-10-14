import numpy as np,pandas as pd
def save(te,col,name):
    s=pd.read_csv('../data/sample_submission.csv')[['Index']]
    o=s.merge(te[['Index',col]].rename(columns={col:'target'}),on='Index',how='left',validate='one_to_one')
    assert o.target.notna().all() and len(o)==11013
    o['target']=o.target.clip(0,110); o.to_csv(f'../submissions/{name}.csv',index=False); print('saved',name)
def polyproj(d,col,deg,lam):
    out=np.empty(len(d))
    for idx in d.groupby('patient_id',sort=False).indices.values():
        t=d.age.values[idx]; t=t-t.mean(); A=np.vander(t,deg+1,increasing=True); z=d[col].values[idx]
        R=np.diag([0]+[lam]*deg); b=np.linalg.solve(A.T@A+R+1e-9*np.eye(deg+1),A.T@z); out[idx]=A@b
    return out
