"""Push a Skore Hub report for the final blend submission.

The report is published to:
    https://skore.probabl.ai/iro-team/bobathon-esilv/reports

Usage (from code/):
    python skore_report.py <submission_name>

    <submission_name> must match the OUT argument given to final_blend3.py
    (e.g. "53_blend_nn12").

Prerequisites:
    1. Run final_blend3.py first (writes blend_meta.pkl).
    2. Authenticate once with:  skore hub api-key generate --workspace iro-team

Environment variable alternative:
    SKORE_HUB_API_KEY=<your-key> python skore_report.py 53_blend_nn12
"""
import sys, pickle, pathlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from skore import EstimatorReport, CrossValidationReport, Project
from skore._reports.cross_validation.report import _BUILTIN_CHECKS
from skore._reports.base import _BaseReport

# ── artefacts ────────────────────────────────────────────────────────────────
META_PATH = pathlib.Path("blend_meta.pkl")
if not META_PATH.exists():
    raise FileNotFoundError("blend_meta.pkl not found – run final_blend3.py first.")

meta    = pickle.load(open(META_PATH, "rb"))
folds   = pickle.load(open("folds.pkl", "rb"))
G       = pd.read_pickle("tr_gen.pkl")[["Index", "target"]]
sub_name = sys.argv[1] if len(sys.argv) > 1 else meta["submission"]

# Load the blend OOF predictions that final_blend3.py computed.
# blend_meta stores weights but not oof directly; reload the component oof
# from the same data final_blend3.py used.
C        = meta["components"]
w        = np.array([meta["weights"][c] for c in C])
y        = G.target.values

# Reconstruct OOF blend the same way final_blend3.py does: read each component.
# We need the OOF columns → load tr_gen + pair + ens3 preds exactly as
# final_blend3.py does, but only the blend result column matters here.
import glob as _glob
Gf = pd.read_pickle("tr_gen.pkl")[["Index", "patient_id", "age", "target", "mgen_q"]]
for k, (o, _) in pickle.load(open("ens3_preds.pkl", "rb")).items():
    Gf["e_" + k] = np.asarray(o)
for f, c in [("pair2", "mpair2_q"), ("pair3", "mpair3_q")]:
    Gf = Gf.merge(pd.read_pickle(f"tr_{f}.pkl")[["Index", c]], on="Index")
tags = sorted(f[2:-7] for f in _glob.glob("nn*_te.pkl"))
for tg in tags:
    Gf = Gf.merge(pd.read_pickle(f"nn{tg}_tr.pkl"), on="Index")
nnc = [f"nn{tg}" for tg in tags]
from post import polyproj
for d in (Gf,):
    d["nnavg"]   = d[nnc].mean(axis=1)
    d["nnavg_q"] = polyproj(d, "nnavg", 2, 0)
oof_blend = Gf[C].values @ w

# ── sklearn wrapper: predict(X) returns X[:,0] ───────────────────────────────
class _BlendRegressor(BaseEstimator, RegressorMixin):
    """Thin sklearn shell; predictions are passed in as X[:,0]."""
    def fit(self, X, y):   self.is_fitted_ = True; return self
    def predict(self, X):  return X[:, 0]

# ── build one EstimatorReport per fold ───────────────────────────────────────
fold_reports = []
for train_idx, test_idx in folds:
    preds_i = oof_blend[test_idx].reshape(-1, 1)
    y_i     = y[test_idx]
    est     = _BlendRegressor().fit(preds_i, y_i)
    fold_reports.append(EstimatorReport(est, X_test=preds_i, y_test=y_i))

# ── assemble CrossValidationReport without re-fitting ────────────────────────
cv = CrossValidationReport.__new__(CrossValidationReport)
_BaseReport.__init__(cv)
cv._initialized_with_data_op = False
cv._ml_task                  = "regression"
cv._pos_label                = None
cv.estimator                 = _BlendRegressor()
cv.learner_                  = _BlendRegressor().fit(oof_blend.reshape(-1, 1), y)
cv._data                     = None
cv._split_indices            = list(folds)
cv.n_jobs                    = None
cv._splitter                 = None
cv._checks_registry          = list(_BUILTIN_CHECKS)
cv.reports_                  = fold_reports

# sanity check
rmse_df = cv.metrics.rmse()
print("CrossValidationReport RMSE per fold:")
print(rmse_df.to_string())
print(f"\nOOF RMSE (global NNLS)  : {meta['rmse_nnls']:.4f}")
print(f"OOF RMSE (fold-honest)  : {meta['rmse_honest']:.4f}")

# ── push to Skore Hub ─────────────────────────────────────────────────────────
project = Project(
    name      = "bobathon-esilv",
    mode      = "hub",
    workspace = "iro-team",
)
project.put(sub_name, cv)
print(f"\nReport pushed → https://skore.probabl.ai/iro-team/bobathon-esilv/reports")
