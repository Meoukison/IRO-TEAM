import numpy as np, pandas as pd
def design(d, kind):
    t=d.ledd.notna().astype(float).values; L=d.ledd.fillna(0).values/600
    if kind=='off':
        k=d.time_since_intake_off.notna().astype(float).values; T=d.time_since_intake_off.fillna(0).values
        return np.c_[np.ones(len(d)), t, k, L, k*T]
    T=d.time_since_intake_on.fillna(d.time_since_intake_on.median()).values; kn=d.time_since_intake_on.isna().astype(float).values
    return np.c_[np.ones(len(d)), t, L, T, kn, d.dur.fillna(0).values/10]
def fit_inv(tr, kind):
    """meas = (D@a)*target + D@c ; returns params via least squares"""
    m=tr[kind].notna().values; D=design(tr[m],kind); y=tr[kind].values[m]; T=tr.target.values[m]
    A=np.c_[D*T[:,None], D]; p,*_=np.linalg.lstsq(A,y,rcond=None); return p
def invert(d, kind, p):
    D=design(d,kind); k=D.shape[1]; a=D@p[:k]; c=D@p[k:]
    return (d[kind].values-c)/np.clip(a,0.2,None)
