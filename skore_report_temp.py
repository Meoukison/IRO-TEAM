"""Publish a lightweight, temporary Skore report from cached OOF predictions.

Run from this directory with:
    python skore_report_temp.py

The report uses the already computed out-of-fold ``mgen_q`` predictions and
does not train or regenerate any model artifacts.
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from skore import CrossValidationReport, EstimatorReport, Project, login
from skore._reports.base import _BaseReport
from skore._reports.cross_validation.report import _BUILTIN_CHECKS
from skore._utils.skrub import to_learner


class CachedPredictionRegressor(BaseEstimator, RegressorMixin):
    """Sklearn wrapper for cached predictions passed as the sole feature."""

    def fit(self, X, y):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0]


data = pd.read_pickle("tr_gen.pkl")[["target", "mgen_q"]]
folds = pickle.load(open("folds.pkl", "rb"))
y = data["target"].to_numpy(dtype=float)
predictions = data["mgen_q"].to_numpy(dtype=float)
if not np.isfinite(y).all() or not np.isfinite(predictions).all():
    raise ValueError("Cached OOF targets or predictions contain non-finite values")

fold_reports = []
for train_idx, test_idx in folds:
    X_train = predictions[train_idx].reshape(-1, 1)
    y_train = y[train_idx]
    X_test = predictions[test_idx].reshape(-1, 1)
    y_test = y[test_idx]
    estimator = CachedPredictionRegressor()
    fold_reports.append(
        EstimatorReport(
            estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
    )

report = CrossValidationReport.__new__(CrossValidationReport)
_BaseReport.__init__(report)
report._initialized_with_data_op = False
report._ml_task = "regression"
report._pos_label = None
report.estimator = CachedPredictionRegressor()
report.learner_ = to_learner(
    CachedPredictionRegressor().fit(predictions.reshape(-1, 1), y)
)
report._data = {
    "_skrub_X": predictions.reshape(-1, 1),
    "_skrub_y": y,
}
report._split_indices = list(folds)
report.n_jobs = None
report._splitter = None
report._checks_registry = list(_BUILTIN_CHECKS)
report.reports_ = fold_reports

print("Temporary report from cached mgen_q OOF predictions")
print(report.metrics.rmse().to_string())
print(f"Global OOF RMSE: {np.sqrt(np.mean((predictions - y) ** 2)):.5f}")

login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace="iro-team")
project.put("53_temp_mgen_q", report)
print("Published: https://skore.probabl.ai/iro-team/bobathon-esilv/reports")
