import numpy as np, pandas as pd
def curve(d, col, sig2, prior_var=(1e4, 1.0, 0.02), prior_mean=(0,2.5,0), deg=2):
    """Bayesian polynomial fit per patient of d[col] vs centered age; returns fitted values."""
    out=np.empty(len(d)); slope=np.empty(len(d))
    pv=np.array(prior_var[:deg+1]); pm=np.array(prior_mean[:deg+1])
    for idx in d.groupby('patient_id').indices.values():
        t=d.age.values[idx]; t=t-t.mean(); z=d[col].values[idx]
        A=np.vander(t,deg+1,increasing=True)
        P=np.diag(1/pv); prec=A.T@A/sig2+P; b=np.linalg.solve(prec,A.T@z/sig2+P@pm)
        out[idx]=A@b; slope[idx]=b[1] if deg>0 else 0
    return out,slope
