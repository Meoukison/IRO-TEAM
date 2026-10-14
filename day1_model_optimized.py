"""Patient-safe stacked model with aligned first-/second-stage predictions.

Run from the project root: ``python day1_model_optimized.py``.
The script performs nested patient-grouped stacking, validates the blend, then
uses the three outer-fold models as a test-time ensemble (no full-data refit).
"""

import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SUBMISSIONS = ROOT / "submissions"
SUBMISSIONS.mkdir(exist_ok=True)
SEED = 42
# Uses all logical cores by default. Set MODEL_N_JOBS=2 (or another number)
# to limit CPU use on a laptop.
N_JOBS = max(1, int(os.getenv("MODEL_N_JOBS", str(os.cpu_count() or 1))))
N_FOLDS = 5
N_INNER_FOLDS = 4

BASE_PARAMS = dict(
    objective="regression_l2",
    learning_rate=0.06,
    num_leaves=63,
    min_child_samples=100,
    colsample_bytree=0.7,
    subsample=0.8,
    subsample_freq=1,
    reg_lambda=10.0,
    max_bin=127,
    force_col_wise=True,
    n_estimators=1500,
    random_state=SEED,
    n_jobs=N_JOBS,
    verbosity=-1,
)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create features from visit measurements and each patient's timeline."""
    df = frame.sort_values(["patient_id", "age"]).reset_index(drop=True).copy()
    groups = df.groupby("patient_id", sort=False)

    for col in ("off", "on", "ledd"):
        filled = np.full(len(df), np.nan, dtype=float)
        nearest_observed_age_gap = np.full(len(df), np.nan, dtype=float)
        interpolation_span = np.full(len(df), np.nan, dtype=float)
        robust_slope = np.full(len(df), np.nan, dtype=float)
        for positions in groups.indices.values():
            positions = np.asarray(positions)
            values = df.loc[positions, col].to_numpy(dtype=float)
            ages = df.loc[positions, "age"].to_numpy(dtype=float)
            observed = np.isfinite(values)
            if observed.any():
                observed_ages = ages[observed]
                observed_values = values[observed]
                filled[positions] = np.interp(ages, observed_ages, observed_values)

                left_pos = np.searchsorted(observed_ages, ages, side="right") - 1
                right_pos = np.searchsorted(observed_ages, ages, side="left")
                left_gap = np.full(len(ages), np.inf)
                right_gap = np.full(len(ages), np.inf)
                has_left = left_pos >= 0
                has_right = right_pos < len(observed_ages)
                left_gap[has_left] = ages[has_left] - observed_ages[left_pos[has_left]]
                right_gap[has_right] = observed_ages[right_pos[has_right]] - ages[has_right]
                nearest_observed_age_gap[positions] = np.minimum(left_gap, right_gap)
                both_sides = has_left & has_right
                interpolation_span[positions[both_sides]] = (
                    observed_ages[right_pos[both_sides]] - observed_ages[left_pos[both_sides]]
                )
                interpolation_span[positions[~both_sides]] = np.minimum(
                    left_gap[~both_sides], right_gap[~both_sides]
                )

                if observed.sum() >= 2:
                    pair_i, pair_j = np.triu_indices(observed.sum(), k=1)
                    age_delta = observed_ages[pair_j] - observed_ages[pair_i]
                    valid_pairs = age_delta != 0
                    slopes = (observed_values[pair_j[valid_pairs]] - observed_values[pair_i[valid_pairs]]) / age_delta[valid_pairs]
                    if len(slopes):
                        robust_slope[positions] = np.median(slopes)
        df[f"{col}_f"] = filled

        observed_by_patient = df.groupby("patient_id", sort=False)[col]
        observed_count = observed_by_patient.transform("count")
        patient_size = observed_by_patient.transform("size")
        df[f"pobs_count_{col}"] = observed_count
        df[f"pobs_frac_{col}"] = observed_count / patient_size.clip(lower=1)
        df[f"pmiss_frac_{col}"] = 1.0 - df[f"pobs_frac_{col}"]
        df[f"{col}_nearest_obs_age_gap"] = nearest_observed_age_gap
        df[f"{col}_interp_span"] = interpolation_span
        df[f"robust_slope_{col}"] = robust_slope

        patient_values = df.groupby("patient_id", sort=False)[f"{col}_f"]
        for stat in ("mean", "std", "min", "max"):
            df[f"p{stat}_{col}"] = patient_values.transform(stat)
        df[f"{col}_missing"] = df[col].isna().astype("int8")

        by_patient = df.groupby("patient_id", sort=False)[f"{col}_f"]
        for lag in (1, 2, 3):
            df[f"{col}_lag{lag}"] = by_patient.shift(lag)
        for lead in (1, 2, 3):
            df[f"{col}_lead{lead}"] = by_patient.shift(-lead)
        df[f"{col}_diff1"] = df[f"{col}_f"] - df[f"{col}_lag1"]

        # Centered three-visit local mean, using measurements only.
        df[f"{col}_roll3"] = by_patient.transform(
            lambda s: s.rolling(window=3, center=True, min_periods=1).mean()
        )

    groups = df.groupby("patient_id", sort=False)
    age_mean = groups["age"].transform("mean")
    age_centered = df["age"] - age_mean
    age_ss = (age_centered**2).groupby(df["patient_id"], sort=False).transform("sum")
    df["dt_prev"] = groups["age"].diff()
    df["dt_next"] = -groups["age"].diff(-1)
    df["t_since_first"] = df["age"] - groups["age"].transform("min")
    df["t_to_last"] = groups["age"].transform("max") - df["age"]
    df["duration"] = df["age"] - df["age_at_diagnosis"]
    df["visit_idx"] = groups.cumcount()
    df["n_visits"] = groups["Index"].transform("size")
    df["visit_frac"] = df["visit_idx"] / (df["n_visits"] - 1).clip(lower=1)
    df["on_off"] = df["on_f"] - df["off_f"]
    df["delta_off"] = df["off_f"] - df["pmean_off"]
    df["delta_on"] = df["on_f"] - df["pmean_on"]

    for col in ("off", "on", "ledd"):
        centered = df[f"{col}_f"] - df[f"pmean_{col}"]
        numerator = (age_centered * centered).groupby(
            df["patient_id"], sort=False
        ).transform("sum")
        df[f"slope_{col}"] = numerator / age_ss.clip(lower=1e-6)

    # The challenge data is synthetic and its stated drug-response mechanism
    # follows levodopa concentration over time. Add compact PK-shaped dose
    # features instead of asking trees to rediscover an exponential curve.
    for state in ("off", "on"):
        time_col = f"time_since_intake_{state}"
        patient_time_median = df.groupby("patient_id", sort=False)[time_col].transform("median")
        fallback_time = 13.8 if state == "off" else 1.6
        elapsed = df[time_col].fillna(patient_time_median).fillna(fallback_time).clip(lower=0)
        measured = df[f"{state}_f"]
        for half_life in (0.5, 1.5, 3.0):
            elimination = np.log(2.0) / half_life
            absorption = 2.5
            peak_time = np.log(absorption / elimination) / (absorption - elimination)
            peak = np.exp(-elimination * peak_time) - np.exp(-absorption * peak_time)
            concentration = (
                np.exp(-elimination * elapsed) - np.exp(-absorption * elapsed)
            ) / peak
            suffix = str(half_life).replace(".", "p")
            df[f"pk_curve_{state}_{suffix}"] = concentration
            df[f"pk_dose_{state}_{suffix}"] = df["ledd_f"] * concentration / 100.0
            df[f"pk_adjusted_{state}_{suffix}"] = measured + df[f"pk_dose_{state}_{suffix}"]

    # Patient-specific levodopa responsiveness estimated only from observed
    # ON/OFF gaps and dose-exposure features, never from the target.
    pk_delta = df["pk_dose_on_1p5"] - df["pk_dose_off_1p5"]
    sensitivity = (df["off_f"] - df["on_f"]) / pk_delta.where(pk_delta.abs() > 0.1)
    df["response_per_pk"] = sensitivity.clip(-10.0, 10.0)
    response_by_patient = df.groupby("patient_id", sort=False)["response_per_pk"]
    df["pat_response_median"] = response_by_patient.transform("median")
    df["pat_response_mean"] = response_by_patient.transform("mean")
    df["pat_response_std"] = response_by_patient.transform("std")

    df["gene_enc"] = df["gene"].map(
        {"No Mutation": 0, "LRRK2+": 1, "GBA+": 2, "OTHER+": 3}
    ).fillna(-1).astype("int8")
    df["cohort_enc"] = (df["cohort"] == "B").astype("int8")
    return df


def add_neighbor_preds(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize first-stage predictions within each complete patient history."""
    df = frame.copy()
    grouped = df.groupby("patient_id", sort=False)["pred1"]
    for lag in (1, 2, 3):
        df[f"pred1_prev{lag}"] = grouped.shift(lag)
        df[f"pred1_next{lag}"] = grouped.shift(-lag)
    df["pred1_pmean"] = grouped.transform("mean")
    df["pred1_pstd"] = grouped.transform("std")
    df["pred1_dev"] = df["pred1"] - df["pred1_pmean"]
    return df


def smooth_patient_predictions(frame: pd.DataFrame, predictions: np.ndarray, window: int) -> np.ndarray:
    """Smooth predictions across nearby visits, without using target values."""
    tmp = pd.DataFrame({"patient_id": frame["patient_id"].to_numpy(), "pred": predictions})
    return tmp.groupby("patient_id", sort=False)["pred"].transform(
        lambda values: values.rolling(window=window, center=True, min_periods=1).mean()
    ).to_numpy()


def fit_simplex_weights(predictions: np.ndarray, target: np.ndarray, initial: np.ndarray) -> np.ndarray:
    """Fit nonnegative ensemble weights that sum to one."""
    errors = predictions - target[:, None]
    covariance = errors.T @ errors / max(1, len(target))
    n_models = predictions.shape[1]
    result = minimize(
        fun=lambda weights: float(weights @ covariance @ weights),
        x0=np.asarray(initial, dtype=float),
        jac=lambda weights: 2.0 * covariance @ weights,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{
            "type": "eq",
            "fun": lambda weights: float(weights.sum() - 1.0),
            "jac": lambda weights: np.ones(n_models),
        }],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    weights = np.clip(result.x if result.success else initial, 0.0, 1.0)
    return weights / weights.sum()


def apply_segment_weights(
    predictions: np.ndarray, segments: np.ndarray, weights_by_segment: dict[str, np.ndarray]
) -> np.ndarray:
    """Blend model predictions with a separate weight vector per segment."""
    blended = np.zeros(len(predictions), dtype=float)
    for segment, weights in weights_by_segment.items():
        mask = segments == segment
        blended[mask] = predictions[mask] @ weights
    return blended


def main() -> None:
    x_train = pd.read_csv(DATA / "X_train.csv")
    y_train = pd.read_csv(DATA / "y_train.csv")
    x_test = pd.read_csv(DATA / "X_test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    train = add_features(x_train.merge(y_train, on="Index", validate="one_to_one"))
    test = add_features(x_test)

    features = [
        "sexM", "age", "age_at_diagnosis", "duration", "visit_idx", "n_visits", "visit_frac",
        "t_since_first", "t_to_last", "dt_prev", "dt_next",
        "off_f", "on_f", "ledd_f", "pmean_off", "pstd_off", "pmin_off", "pmax_off",
        "pmean_on", "pstd_on", "pmin_on", "pmax_on", "pmean_ledd", "pstd_ledd",
        "pmin_ledd", "pmax_ledd", "on_off", "delta_off", "delta_on",
        "slope_off", "slope_on", "slope_ledd", "gene_enc", "cohort_enc",
        "pobs_count_off", "pobs_frac_off", "pmiss_frac_off",
        "pobs_count_on", "pobs_frac_on", "pmiss_frac_on",
        "pobs_count_ledd", "pobs_frac_ledd", "pmiss_frac_ledd",
        "off_nearest_obs_age_gap", "off_interp_span", "robust_slope_off",
        "on_nearest_obs_age_gap", "on_interp_span", "robust_slope_on",
        "ledd_nearest_obs_age_gap", "ledd_interp_span", "robust_slope_ledd",
        "time_since_intake_on", "time_since_intake_off",
        "off_missing", "on_missing", "ledd_missing",
    ]
    for col in ("off", "on", "ledd"):
        features.extend([f"{col}_lag{k}" for k in (1, 2, 3)])
        features.extend([f"{col}_lead{k}" for k in (1, 2, 3)])
        features.extend([f"{col}_diff1", f"{col}_roll3"])
    for state in ("off", "on"):
        for half_life in ("0p5", "1p5", "3p0"):
            features.extend([
                f"pk_curve_{state}_{half_life}",
                f"pk_dose_{state}_{half_life}",
                f"pk_adjusted_{state}_{half_life}",
            ])
    features.extend(["response_per_pk", "pat_response_median", "pat_response_mean", "pat_response_std"])

    neighbor_features = [
        *(f"pred1_prev{k}" for k in (1, 2, 3)),
        *(f"pred1_next{k}" for k in (1, 2, 3)),
        "pred1", "pred1_pmean", "pred1_pstd", "pred1_dev",
    ]
    stack_features = features + neighbor_features
    x = train[features]
    y = train["target"].to_numpy()
    patient_ids = train["patient_id"].to_numpy()

    oof_lgb = np.zeros(len(train), dtype=float)
    oof_lgb_off_residual = np.zeros(len(train), dtype=float)
    oof_hist = np.zeros(len(train), dtype=float)
    oof_xgb = np.zeros(len(train), dtype=float)
    oof_xgb_small = np.zeros(len(train), dtype=float)
    oof_cat = np.zeros(len(train), dtype=float)
    outer_fold_id = np.full(len(train), -1, dtype=np.int8)
    test_lgb_by_fold = []
    test_lgb_off_residual_by_fold = []
    test_hist_by_fold = []
    test_xgb_by_fold = []
    test_xgb_small_by_fold = []
    test_cat_by_fold = []
    outer_cv = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    print(f"Stacking OOF par patient : {N_FOLDS} plis; LightGBM utilise {N_JOBS} cœur(s).", flush=True)

    for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(x, y, patient_ids), start=1):
        outer_fold_id[va_idx] = fold - 1
        outer_train = train.iloc[tr_idx].copy().reset_index(drop=True)
        outer_valid = train.iloc[va_idx].copy().reset_index(drop=True)
        outer_test = test.copy()
        inner_cv = GroupKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED + fold)
        pred1_train = np.zeros(len(outer_train), dtype=float)
        first_models = []

        # Each training patient's first-stage prediction comes from a model
        # that excluded every visit of that patient. Outer validation patients
        # are excluded from every first-stage fit in this fold.
        for inner_tr, inner_va in inner_cv.split(
            outer_train[features], outer_train["target"], outer_train["patient_id"]
        ):
            first = lgb.LGBMRegressor(**BASE_PARAMS)
            first.fit(outer_train.iloc[inner_tr][features], outer_train.iloc[inner_tr]["target"])
            pred1_train[inner_va] = first.predict(outer_train.iloc[inner_va][features])
            first_models.append(first)

        outer_train["pred1"] = pred1_train
        outer_train = add_neighbor_preds(outer_train)

        # Stage 2 is trained on one cross-fitted stage-1 prediction per row.
        # Keep that same prediction distribution at inference: apply each
        # inner stage-1 model separately through stage 2, then average complete
        # predictions. Averaging stage 1 first makes its features too smooth.
        valid_views = []
        test_views = []
        for first in first_models:
            valid_view = outer_valid.copy()
            valid_view["pred1"] = first.predict(valid_view[features])
            valid_views.append(add_neighbor_preds(valid_view))

            test_view = outer_test.copy()
            test_view["pred1"] = first.predict(test_view[features])
            test_views.append(add_neighbor_preds(test_view))

        residual_train = outer_train["target"].to_numpy() - outer_train["pred1"].to_numpy()
        residual_valid = outer_valid["target"].to_numpy() - valid_views[0]["pred1"].to_numpy()

        second_lgb = lgb.LGBMRegressor(**{**BASE_PARAMS, "n_estimators": 900})
        second_lgb.fit(
            outer_train[stack_features], residual_train,
            eval_X=valid_views[0][stack_features], eval_y=residual_valid,
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        pred_valid_lgb = np.mean([
            view["pred1"].to_numpy() + second_lgb.predict(view[stack_features])
            for view in valid_views
        ], axis=0)
        oof_lgb[va_idx] = pred_valid_lgb

        # Alternative stage-2 target: correct the interpolated OFF estimate.
        # Patients with no OFF observations use the stage-1 prediction as a
        # safe baseline, so this remains defined for every row.
        outer_train["off_base_est"] = outer_train["off_f"].fillna(outer_train["pred1"])
        off_residual_train = outer_train["target"].to_numpy() - outer_train["off_base_est"].to_numpy()
        valid_off_view = valid_views[0].copy()
        valid_off_view["off_base_est"] = valid_off_view["off_f"].fillna(valid_off_view["pred1"])
        off_residual_valid = outer_valid["target"].to_numpy() - valid_off_view["off_base_est"].to_numpy()
        off_stage2_features = stack_features + ["off_base_est"]
        second_lgb_off = lgb.LGBMRegressor(**{**BASE_PARAMS, "n_estimators": 900})
        second_lgb_off.fit(
            outer_train[off_stage2_features], off_residual_train,
            eval_X=valid_off_view[off_stage2_features], eval_y=off_residual_valid,
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        valid_off_predictions = []
        test_off_predictions = []
        for view in valid_views:
            off_view = view.copy()
            off_view["off_base_est"] = off_view["off_f"].fillna(off_view["pred1"])
            valid_off_predictions.append(
                off_view["off_base_est"].to_numpy()
                + second_lgb_off.predict(off_view[off_stage2_features])
            )
        for view in test_views:
            off_view = view.copy()
            off_view["off_base_est"] = off_view["off_f"].fillna(off_view["pred1"])
            test_off_predictions.append(
                off_view["off_base_est"].to_numpy()
                + second_lgb_off.predict(off_view[off_stage2_features])
            )
        pred_valid_lgb_off = np.mean(valid_off_predictions, axis=0)
        oof_lgb_off_residual[va_idx] = pred_valid_lgb_off

        second_hist = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=300,
            max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=5.0,
            early_stopping=False, random_state=SEED,
        )
        second_hist.fit(outer_train[stack_features], residual_train)
        pred_valid_hist = np.mean([
            view["pred1"].to_numpy() + second_hist.predict(view[stack_features])
            for view in valid_views
        ], axis=0)
        oof_hist[va_idx] = pred_valid_hist

        second_xgb = XGBRegressor(
            objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
            n_estimators=800, max_depth=7, learning_rate=0.08,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=10.0, max_bin=127, n_jobs=N_JOBS,
            random_state=SEED, verbosity=0,
        )
        second_xgb.fit(outer_train[stack_features], residual_train, verbose=False)
        pred_valid_xgb = np.mean([
            view["pred1"].to_numpy() + second_xgb.predict(view[stack_features])
            for view in valid_views
        ], axis=0)
        oof_xgb[va_idx] = pred_valid_xgb

        second_xgb_small = XGBRegressor(
            objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
            n_estimators=1200, max_depth=5, learning_rate=0.05,
            min_child_weight=15, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=15.0, max_bin=127, n_jobs=N_JOBS,
            random_state=SEED + 1, verbosity=0,
        )
        second_xgb_small.fit(outer_train[stack_features], residual_train, verbose=False)
        pred_valid_xgb_small = np.mean([
            view["pred1"].to_numpy() + second_xgb_small.predict(view[stack_features])
            for view in valid_views
        ], axis=0)
        oof_xgb_small[va_idx] = pred_valid_xgb_small

        second_cat = CatBoostRegressor(
            iterations=800, depth=7, learning_rate=0.08,
            loss_function="RMSE", eval_metric="RMSE", l2_leaf_reg=8.0,
            random_seed=SEED, thread_count=N_JOBS, verbose=False,
            allow_writing_files=False,
        )
        second_cat.fit(outer_train[stack_features], residual_train, verbose=False)
        pred_valid_cat = np.mean([
            view["pred1"].to_numpy() + second_cat.predict(view[stack_features])
            for view in valid_views
        ], axis=0)
        oof_cat[va_idx] = pred_valid_cat

        # Each first-stage model passes separately through stage 2; average
        # complete predictions only after the residual correction.
        test_lgb_by_fold.append(np.mean([
            view["pred1"].to_numpy() + second_lgb.predict(view[stack_features])
            for view in test_views
        ], axis=0))
        test_lgb_off_residual_by_fold.append(np.mean(test_off_predictions, axis=0))
        test_hist_by_fold.append(np.mean([
            view["pred1"].to_numpy() + second_hist.predict(view[stack_features])
            for view in test_views
        ], axis=0))
        test_xgb_by_fold.append(np.mean([
            view["pred1"].to_numpy() + second_xgb.predict(view[stack_features])
            for view in test_views
        ], axis=0))
        test_xgb_small_by_fold.append(np.mean([
            view["pred1"].to_numpy() + second_xgb_small.predict(view[stack_features])
            for view in test_views
        ], axis=0))
        test_cat_by_fold.append(np.mean([
            view["pred1"].to_numpy() + second_cat.predict(view[stack_features])
            for view in test_views
        ], axis=0))
        fold_rmse = root_mean_squared_error(y[va_idx], pred_valid_lgb)
        print(
            f"  Pli {fold}/{N_FOLDS}: RMSE LightGBM={fold_rmse:.4f}; "
            f"étape 1={len(first_models)} modèles, étape 2={second_lgb.best_iteration_}",
            flush=True,
        )

    rmse_lgb = root_mean_squared_error(y, oof_lgb)
    rmse_lgb_off_residual = root_mean_squared_error(y, oof_lgb_off_residual)
    rmse_hist = root_mean_squared_error(y, oof_hist)
    rmse_xgb = root_mean_squared_error(y, oof_xgb)
    rmse_xgb_small = root_mean_squared_error(y, oof_xgb_small)
    rmse_cat = root_mean_squared_error(y, oof_cat)
    oof_predictions = np.column_stack([
        oof_lgb, oof_lgb_off_residual, oof_hist, oof_xgb, oof_xgb_small, oof_cat,
    ])
    initial_weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    global_weights = fit_simplex_weights(oof_predictions, y, initial_weights)
    segments = np.select(
        [train["off"].notna() & train["time_since_intake_off"].notna(), train["off"].notna()],
        ["off+délai", "off seul"], default="off manquant",
    )

    # Learn segment-specific blend weights on the other outer folds and score
    # them on the held-out fold. This prevents selecting weights on the same
    # OOF rows whose RMSE is reported.
    meta_alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
    meta_oof = {alpha: np.zeros(len(train), dtype=float) for alpha in meta_alphas}
    segment_names = ("off+délai", "off seul", "off manquant")
    for held_fold in range(N_FOLDS):
        fit_mask = outer_fold_id != held_fold
        valid_mask = outer_fold_id == held_fold
        fold_global = fit_simplex_weights(
            oof_predictions[fit_mask], y[fit_mask], global_weights
        )
        for segment in segment_names:
            fit_segment = fit_mask & (segments == segment)
            valid_segment = valid_mask & (segments == segment)
            fold_segment = fit_simplex_weights(
                oof_predictions[fit_segment], y[fit_segment], fold_global
            )
            for alpha in meta_alphas:
                weights = (1.0 - alpha) * fold_global + alpha * fold_segment
                meta_oof[alpha][valid_segment] = oof_predictions[valid_segment] @ weights

    meta_scores = {
        alpha: root_mean_squared_error(y, prediction)
        for alpha, prediction in meta_oof.items()
    }
    meta_alpha = min(meta_scores, key=meta_scores.get)
    print(
        f"OOF modèles : LGBM={rmse_lgb:.4f}, OFF-résiduel={rmse_lgb_off_residual:.4f}, "
        f"HistGB={rmse_hist:.4f}, XGB7={rmse_xgb:.4f}, XGB5={rmse_xgb_small:.4f}, "
        f"CatBoost={rmse_cat:.4f}; méta-blend CV global={meta_scores[0.0]:.4f}, "
        f"par motif={meta_scores[meta_alpha]:.4f} (alpha={meta_alpha:.2f}).",
        flush=True,
    )

    weights_by_segment = {}
    for segment in segment_names:
        mask = segments == segment
        segment_weights = fit_simplex_weights(
            oof_predictions[mask], y[mask], global_weights
        )
        weights_by_segment[segment] = (
            (1.0 - meta_alpha) * global_weights + meta_alpha * segment_weights
        )
        print(
            f"  {segment}: poids={np.round(weights_by_segment[segment], 3).tolist()}",
            flush=True,
        )

    oof_blend = apply_segment_weights(oof_predictions, segments, weights_by_segment)
    oof_meta_cv = meta_oof[meta_alpha]
    raw_rmse = root_mean_squared_error(y, oof_meta_cv)
    best_smooth_rmse = raw_rmse
    best_smooth = (0.0, 0)
    for window in (3, 5, 7, 9):
        smoothed = smooth_patient_predictions(train, oof_meta_cv, window)
        for alpha in np.linspace(0.0, 1.0, 21):
            candidate = (1.0 - alpha) * oof_meta_cv + alpha * smoothed
            score = root_mean_squared_error(y, candidate)
            if score < best_smooth_rmse:
                best_smooth_rmse = score
                best_smooth = (float(alpha), window)
    print(
        f"Lissage OOF méta-CV : {raw_rmse:.4f} -> {best_smooth_rmse:.4f}; "
        f"alpha={best_smooth[0]:.2f}, fenêtre={best_smooth[1]}.",
        flush=True,
    )

    oof_final = oof_meta_cv
    if best_smooth[0] > 0:
        oof_final = (1.0 - best_smooth[0]) * oof_meta_cv + best_smooth[0] * smooth_patient_predictions(
            train, oof_meta_cv, best_smooth[1]
        )
    print("Erreur OOF par disponibilité de off/délai :", flush=True)
    for segment in ("off+délai", "off seul", "off manquant"):
        mask = segments == segment
        mse_segment = float(np.mean((y[mask] - oof_final[mask]) ** 2))
        contribution = float(np.mean((y[mask] - oof_final[mask]) ** 2)) * float(mask.mean())
        print(
            f"  {segment}: n={int(mask.sum())}, RMSE={mse_segment ** 0.5:.4f}, "
            f"part du MSE total={100 * contribution / root_mean_squared_error(y, oof_final) ** 2:.1f}%",
            flush=True,
        )

    test_predictions = np.column_stack([
        np.mean(test_lgb_by_fold, axis=0),
        np.mean(test_lgb_off_residual_by_fold, axis=0),
        np.mean(test_hist_by_fold, axis=0),
        np.mean(test_xgb_by_fold, axis=0),
        np.mean(test_xgb_small_by_fold, axis=0),
        np.mean(test_cat_by_fold, axis=0),
    ])
    test_segments = np.select(
        [test["off"].notna() & test["time_since_intake_off"].notna(), test["off"].notna()],
        ["off+délai", "off seul"], default="off manquant",
    )
    predictions = np.clip(
        apply_segment_weights(test_predictions, test_segments, weights_by_segment), 0, 110
    )
    if best_smooth[0] > 0:
        test_smoothed = smooth_patient_predictions(test, predictions, best_smooth[1])
        predictions = (1.0 - best_smooth[0]) * predictions + best_smooth[0] * test_smoothed
    submission = sample[["Index"]].merge(
        pd.DataFrame({"Index": test["Index"], "target": predictions}),
        on="Index", how="left", validate="one_to_one",
    )
    if submission["target"].isna().any():
        raise ValueError("Certaines lignes de soumission n'ont pas reçu de prédiction.")

    output = SUBMISSIONS / "21_segment_ensemble.csv"
    if best_smooth_rmse >= 3.55:
        print("Candidat rejeté : RMSE OOF non lissé ≥ 3,55; aucun nouveau CSV écrit.", flush=True)
        return
    submission.to_csv(output, index=False)
    print(f"Soumission créée : {output}", flush=True)


if __name__ == "__main__":
    main()
