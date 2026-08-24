"""
Leakage-controlled two-level stacking for the IGRM data.

Design
------
1. Test1 and Test2 are not loaded until all model selection and fitting are done.
2. Every training sample receives base-learner meta-features from a model that
   was fitted without that sample (strict out-of-fold prediction).
3. In every outer OOF fold, preprocessing, feature ranking, and base-learner
   hyperparameter selection are fitted using the outer-training partition only.
4. A nested whole-stack prediction is also produced for each outer validation
   fold. This is the unbiased training-only estimate of the complete stack.
5. Final base configurations are selected using training-only cross-validation,
   then preprocessing, feature ranking, and base learners are refitted on the
   complete development set. The OOF-trained logistic regression is retained.
6. The two external test sets are opened and evaluated exactly once after a
   model-lock file has been written.

Run from the repository root:
    python code/train_strict_oof.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model_pkl"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILES = {
    "test1": DATA_DIR / "test1.csv",
    "test2": DATA_DIR / "test2.csv",
}

RANDOM_SEED = 20260729
OUTER_SPLITS = 5
INNER_SPLITS = 3
FINAL_TUNING_SPLITS = 5
REPEATS = 1
CLASSIFICATION_THRESHOLD = 0.50
BOOTSTRAP_REPLICATES = 1000

MODEL_ORDER = ["LightGBM", "XGBoost", "RF", "GBDT"]
ID_COLUMNS = ["chr", "pos", "ref", "alt"]
LABEL_COLUMN = "Label"
EXCLUDE_COLUMNS = {
    "chr",
    "pos",
    "ref",
    "alt",
    "Label",
    "label",
    "chrom",
    "Is_Transition",
    "Is_Synonymous",
}

# Kept for schema compatibility. The current train.csv has already removed
# these correlated/redundant columns.
FEATURES_TO_REMOVE = {
    "CADD_GC",
    "CADD_CpG",
    "CADD_cHmmTx",
    "CADD_EncH3K4Me3",
    "CADD_ZooVerPhyloP",
    "MMS_efficiency",
    "Synmall_Generic_deleteriousness_scores_cadd_phred",
    "Synmall_Generic_functional_scores_cadd_splice_phred",
    "GC_content_10bp",
    "has_codon_mutation",
    "EncodeH3K4me2-max",
    "EncodeH3K4me3-max",
    "EncodeH3K9ac-max",
    "EncodeH3K9me3-max",
    "EncodeH3K27ac-max",
    "EncodeH3K36me3-max",
    "EncodeH2AFZ-max",
    "EncodeDNase-max",
    "RemapOverlapCL",
    "MACIE00",
    "RBP_Motif_Hits_Alt",
    "m6A_Motif_Alt",
    "RNA_Mod_Total_Alt",
    "Ribo_Pause_Score_Alt",
    "Ribo_Pause_Codon_Count",
    "Translation_Rate_Alt",
    "CpG_Density",
    "Open_Chromatin_Motifs",
    "Predicted_Accessibility",
    "Is_Transversion",
    "Phys_Delta_Tm",
    "Ctx_GC_Content",
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    n_features: int | None
    params: dict[str, Any]


CANDIDATES: dict[str, list[Candidate]] = {
    "LightGBM": [
        Candidate(
            "lgbm_compact",
            100,
            {
                "n_estimators": 200,
                "max_depth": 7,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 20,
                "reg_lambda": 1.0,
            },
        ),
        Candidate(
            "lgbm_original",
            120,
            {
                "n_estimators": 500,
                "max_depth": 15,
                "learning_rate": 0.05,
                "num_leaves": 50,
                "min_child_samples": 20,
                "reg_lambda": 1.0,
            },
        ),
        Candidate(
            "lgbm_regularized",
            160,
            {
                "n_estimators": 300,
                "max_depth": 10,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 30,
                "reg_alpha": 0.1,
                "reg_lambda": 2.0,
            },
        ),
    ],
    "XGBoost": [
        Candidate(
            "xgb_compact",
            120,
            {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "reg_lambda": 1.0,
            },
        ),
        Candidate(
            "xgb_slow",
            150,
            {
                "n_estimators": 300,
                "max_depth": 8,
                "learning_rate": 0.02,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "reg_lambda": 2.0,
            },
        ),
        Candidate(
            "xgb_regularized",
            180,
            {
                "n_estimators": 400,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 5,
                "reg_lambda": 5.0,
            },
        ),
    ],
    "RF": [
        Candidate(
            "rf_compact",
            180,
            {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
        ),
        Candidate(
            "rf_original_regularized",
            None,
            {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
        ),
        Candidate(
            "rf_depth_regularized",
            None,
            {
                "n_estimators": 400,
                "max_depth": 20,
                "min_samples_split": 4,
                "min_samples_leaf": 2,
                "max_features": 0.5,
            },
        ),
    ],
    "GBDT": [
        Candidate(
            "gbdt_shallow",
            120,
            {
                "n_estimators": 200,
                "max_depth": 2,
                "learning_rate": 0.05,
                "subsample": 0.8,
            },
        ),
        Candidate(
            "gbdt_balanced",
            160,
            {
                "n_estimators": 200,
                "max_depth": 3,
                "learning_rate": 0.10,
                "subsample": 0.8,
            },
        ),
        Candidate(
            "gbdt_original",
            200,
            {
                "n_estimators": 200,
                "max_depth": 3,
                "learning_rate": 0.20,
                "subsample": 1.0,
            },
        ),
    ],
}

META_C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(payload), handle, ensure_ascii=False, indent=2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(*parts: int) -> int:
    seed = RANDOM_SEED
    for part in parts:
        seed = (seed * 1664525 + int(part) + 1013904223) % (2**31 - 1)
    return int(seed)


def build_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )


def fit_preprocessing(
    X_fit: np.ndarray, X_apply: np.ndarray
) -> tuple[VarianceThreshold, Pipeline, np.ndarray, np.ndarray]:
    variance_filter = VarianceThreshold(threshold=0.0)
    X_fit_var = variance_filter.fit_transform(X_fit)
    X_apply_var = variance_filter.transform(X_apply)
    preprocessor = build_preprocessor()
    X_fit_processed = preprocessor.fit_transform(X_fit_var)
    X_apply_processed = preprocessor.transform(X_apply_var)
    return variance_filter, preprocessor, X_fit_processed, X_apply_processed


def build_model(family: str, params: dict[str, Any], seed: int) -> BaseEstimator:
    if family == "LightGBM":
        return LGBMClassifier(
            random_state=seed,
            verbosity=-1,
            n_jobs=-1,
            deterministic=True,
            force_col_wise=True,
            **params,
        )
    if family == "XGBoost":
        return XGBClassifier(
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
    if family == "RF":
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    if family == "GBDT":
        return GradientBoostingClassifier(random_state=seed, **params)
    raise ValueError(f"Unknown model family: {family}")


def build_ranker(family: str, seed: int) -> BaseEstimator:
    if family == "LightGBM":
        return LGBMClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            verbosity=-1,
            n_jobs=-1,
            deterministic=True,
            force_col_wise=True,
        )
    if family == "XGBoost":
        return XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
            tree_method="hist",
        )
    if family == "RF":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if family == "GBDT":
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=seed,
        )
    raise ValueError(f"Unknown model family: {family}")


def feature_ranking(
    family: str, X: np.ndarray, y: np.ndarray, seed: int
) -> np.ndarray:
    ranker = build_ranker(family, seed)
    ranker.fit(X, y)
    importances = np.asarray(ranker.feature_importances_, dtype=float)
    if importances.shape[0] != X.shape[1]:
        raise RuntimeError(f"{family} returned an invalid importance vector.")
    return np.argsort(-importances, kind="mergesort")


def selected_indices(candidate: Candidate, ranking: np.ndarray) -> np.ndarray:
    count = len(ranking) if candidate.n_features is None else candidate.n_features
    return np.asarray(ranking[: min(int(count), len(ranking))], dtype=int)


def tune_family(
    X: np.ndarray,
    y: np.ndarray,
    family: str,
    n_splits: int,
    seed: int,
    stage: str,
    tuning_rows: list[dict[str, Any]],
) -> Candidate:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: dict[str, list[tuple[float, float]]] = {
        candidate.candidate_id: [] for candidate in CANDIDATES[family]
    }

    for fold_index, (fit_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        _, _, X_fit, X_val = fit_preprocessing(X[fit_idx], X[val_idx])
        ranking = feature_ranking(
            family, X_fit, y[fit_idx], derive_seed(seed, fold_index, 11)
        )
        for candidate_index, candidate in enumerate(CANDIDATES[family], start=1):
            indices = selected_indices(candidate, ranking)
            model = build_model(
                family,
                candidate.params,
                derive_seed(seed, fold_index, candidate_index, 29),
            )
            model.fit(X_fit[:, indices], y[fit_idx])
            probability = model.predict_proba(X_val[:, indices])[:, 1]
            auc_value = roc_auc_score(y[val_idx], probability)
            aupr_value = average_precision_score(y[val_idx], probability)
            scores[candidate.candidate_id].append((auc_value, aupr_value))
            tuning_rows.append(
                {
                    "stage": stage,
                    "family": family,
                    "candidate_id": candidate.candidate_id,
                    "cv_fold": fold_index,
                    "n_features": len(indices),
                    "AUC": auc_value,
                    "AUPR": aupr_value,
                }
            )

    summary = []
    by_id = {candidate.candidate_id: candidate for candidate in CANDIDATES[family]}
    for candidate_id, fold_scores in scores.items():
        auc_values = [item[0] for item in fold_scores]
        aupr_values = [item[1] for item in fold_scores]
        summary.append(
            (
                -float(np.mean(auc_values)),
                -float(np.mean(aupr_values)),
                float(np.std(auc_values)),
                candidate_id,
            )
        )
    summary.sort()
    return by_id[summary[0][3]]


def tune_meta_c(
    meta_X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[float, list[dict[str, float]]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows: list[dict[str, float]] = []
    for c_value in META_C_GRID:
        fold_auc: list[float] = []
        fold_aupr: list[float] = []
        for fit_idx, val_idx in cv.split(meta_X, y):
            model = LogisticRegression(
                C=c_value,
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                random_state=seed,
            )
            model.fit(meta_X[fit_idx], y[fit_idx])
            probability = model.predict_proba(meta_X[val_idx])[:, 1]
            fold_auc.append(roc_auc_score(y[val_idx], probability))
            fold_aupr.append(average_precision_score(y[val_idx], probability))
        rows.append(
            {
                "C": c_value,
                "mean_AUC": float(np.mean(fold_auc)),
                "std_AUC": float(np.std(fold_auc)),
                "mean_AUPR": float(np.mean(fold_aupr)),
            }
        )
    rows.sort(key=lambda row: (-row["mean_AUC"], -row["mean_AUPR"], row["C"]))
    return float(rows[0]["C"]), rows


def fixed_config_oof(
    X: np.ndarray,
    y: np.ndarray,
    selected: dict[str, Candidate],
    n_splits: int,
    seed: int,
) -> np.ndarray:
    meta_X = np.full((len(y), len(MODEL_ORDER)), np.nan, dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_index, (fit_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        _, _, X_fit, X_val = fit_preprocessing(X[fit_idx], X[val_idx])
        for model_index, family in enumerate(MODEL_ORDER):
            candidate = selected[family]
            ranking = feature_ranking(
                family, X_fit, y[fit_idx], derive_seed(seed, fold_index, model_index, 41)
            )
            indices = selected_indices(candidate, ranking)
            model = build_model(
                family,
                candidate.params,
                derive_seed(seed, fold_index, model_index, 43),
            )
            model.fit(X_fit[:, indices], y[fit_idx])
            meta_X[val_idx, model_index] = model.predict_proba(
                X_val[:, indices]
            )[:, 1]
    if np.isnan(meta_X).any():
        raise RuntimeError("Inner OOF matrix contains missing predictions.")
    return meta_X


def fit_outer_models_and_predict(
    X_fit_raw: np.ndarray,
    y_fit: np.ndarray,
    X_apply_raw: np.ndarray,
    selected: dict[str, Candidate],
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    _, _, X_fit, X_apply = fit_preprocessing(X_fit_raw, X_apply_raw)
    predictions = np.empty((len(X_apply_raw), len(MODEL_ORDER)), dtype=float)
    details: list[dict[str, Any]] = []
    for model_index, family in enumerate(MODEL_ORDER):
        candidate = selected[family]
        ranking = feature_ranking(
            family, X_fit, y_fit, derive_seed(seed, model_index, 51)
        )
        indices = selected_indices(candidate, ranking)
        model = build_model(
            family, candidate.params, derive_seed(seed, model_index, 53)
        )
        model.fit(X_fit[:, indices], y_fit)
        predictions[:, model_index] = model.predict_proba(
            X_apply[:, indices]
        )[:, 1]
        details.append(
            {
                "family": family,
                "candidate_id": candidate.candidate_id,
                "n_features": len(indices),
            }
        )
    return predictions, details


def strict_nested_oof(
    X: np.ndarray,
    y: np.ndarray,
    tuning_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    base_oof = np.full((len(y), len(MODEL_ORDER)), np.nan, dtype=float)
    nested_stack_probability = np.full(len(y), np.nan, dtype=float)
    fold_assignment = np.full(len(y), -1, dtype=int)
    fold_records: list[dict[str, Any]] = []

    outer_cv = StratifiedKFold(
        n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_SEED
    )
    for outer_fold, (fit_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        fold_started = time.time()
        print(
            f"\n[OOF] Outer fold {outer_fold}/{OUTER_SPLITS}: "
            f"fit={len(fit_idx)}, validation={len(val_idx)}",
            flush=True,
        )
        selected: dict[str, Candidate] = {}
        for model_index, family in enumerate(MODEL_ORDER):
            selected[family] = tune_family(
                X[fit_idx],
                y[fit_idx],
                family,
                INNER_SPLITS,
                derive_seed(outer_fold, model_index, 61),
                f"outer_{outer_fold}_inner_tuning",
                tuning_rows,
            )
            print(
                f"  {family}: {selected[family].candidate_id}",
                flush=True,
            )

        # OOF predictions inside the outer-training partition train the
        # fold-specific meta learner without in-sample base probabilities.
        inner_meta_X = fixed_config_oof(
            X[fit_idx],
            y[fit_idx],
            selected,
            INNER_SPLITS,
            derive_seed(outer_fold, 67),
        )
        fold_meta_c, fold_meta_search = tune_meta_c(
            inner_meta_X, y[fit_idx], derive_seed(outer_fold, 71)
        )
        fold_meta = LogisticRegression(
            C=fold_meta_c,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=derive_seed(outer_fold, 73),
        )
        fold_meta.fit(inner_meta_X, y[fit_idx])

        outer_probability_matrix, model_details = fit_outer_models_and_predict(
            X[fit_idx],
            y[fit_idx],
            X[val_idx],
            selected,
            derive_seed(outer_fold, 79),
        )
        base_oof[val_idx] = outer_probability_matrix
        nested_stack_probability[val_idx] = fold_meta.predict_proba(
            outer_probability_matrix
        )[:, 1]
        fold_assignment[val_idx] = outer_fold
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "fit_size": len(fit_idx),
                "validation_size": len(val_idx),
                "selected_candidates": {
                    family: selected[family].candidate_id for family in MODEL_ORDER
                },
                "meta_C": fold_meta_c,
                "meta_search": fold_meta_search,
                "models": model_details,
                "elapsed_seconds": time.time() - fold_started,
            }
        )

    if np.isnan(base_oof).any() or np.isnan(nested_stack_probability).any():
        raise RuntimeError("Outer OOF predictions are incomplete.")
    if np.any(fold_assignment < 1):
        raise RuntimeError("At least one training sample has no outer fold.")
    return base_oof, nested_stack_probability, fold_assignment, fold_records


def fit_final_models(
    X: np.ndarray,
    y: np.ndarray,
    base_oof: np.ndarray,
    feature_cols: list[str],
    tuning_rows: list[dict[str, Any]],
    run_dir: Path,
) -> tuple[
    VarianceThreshold,
    Pipeline,
    dict[str, BaseEstimator],
    dict[str, np.ndarray],
    LogisticRegression,
    dict[str, Candidate],
    dict[str, Any],
]:
    print("\n[FINAL] Training-only selection of final base configurations", flush=True)
    selected: dict[str, Candidate] = {}
    for model_index, family in enumerate(MODEL_ORDER):
        selected[family] = tune_family(
            X,
            y,
            family,
            FINAL_TUNING_SPLITS,
            derive_seed(model_index, 83),
            "final_training_only_tuning",
            tuning_rows,
        )
        print(f"  {family}: {selected[family].candidate_id}", flush=True)

    meta_c, meta_search = tune_meta_c(base_oof, y, derive_seed(89))
    meta_model = LogisticRegression(
        C=meta_c,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=derive_seed(97),
    )
    meta_model.fit(base_oof, y)
    print(f"  LogisticRegression: C={meta_c}", flush=True)

    variance_filter = VarianceThreshold(threshold=0.0)
    X_var = variance_filter.fit_transform(X)
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X_var)
    kept_feature_cols = np.asarray(feature_cols, dtype=object)[
        variance_filter.get_support()
    ].tolist()

    models: dict[str, BaseEstimator] = {}
    feature_indices: dict[str, np.ndarray] = {}
    feature_details: dict[str, Any] = {}
    for model_index, family in enumerate(MODEL_ORDER):
        candidate = selected[family]
        ranking = feature_ranking(
            family, X_processed, y, derive_seed(model_index, 101)
        )
        indices = selected_indices(candidate, ranking)
        model = build_model(
            family, candidate.params, derive_seed(model_index, 103)
        )
        model.fit(X_processed[:, indices], y)
        models[family] = model
        feature_indices[family] = indices
        feature_details[family] = {
            "candidate_id": candidate.candidate_id,
            "n_features": len(indices),
            "processed_feature_indices": indices.tolist(),
            "selected_feature_names": [kept_feature_cols[i] for i in indices],
        }

    model_dir = MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(variance_filter, model_dir / "variance_filter.pkl")
    joblib.dump(preprocessor, model_dir / "preprocessor.pkl")
    joblib.dump(feature_cols, model_dir / "feature_cols.pkl")
    joblib.dump(feature_indices, model_dir / "feature_indices.pkl")
    joblib.dump(MODEL_ORDER, model_dir / "model_names.pkl")
    joblib.dump(meta_model, model_dir / "LR_meta_oof.pkl")
    for family, model in models.items():
        joblib.dump(model, model_dir / f"{family}_base.pkl")

    final_selection = {
        family: {
            "candidate_id": selected[family].candidate_id,
            "n_features": selected[family].n_features,
            "params": selected[family].params,
        }
        for family in MODEL_ORDER
    }
    final_selection["LogisticRegression"] = {
        "C": meta_c,
        "search": meta_search,
    }
    write_json(run_dir / "selected_hyperparameters.json", final_selection)
    write_json(run_dir / "selected_features.json", feature_details)
    return (
        variance_filter,
        preprocessor,
        models,
        feature_indices,
        meta_model,
        selected,
        final_selection,
    )


def variant_keys(frame: pd.DataFrame) -> pd.Series:
    chromosome = (
        frame["chr"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"^chr", "", regex=True)
    )
    position = pd.to_numeric(frame["pos"], errors="raise").astype("Int64").astype(str)
    reference = frame["ref"].astype(str).str.strip().str.upper()
    alternate = frame["alt"].astype(str).str.strip().str.upper()
    return chromosome + "|" + position + "|" + reference + "|" + alternate


def dataset_summary(
    name: str, frame: pd.DataFrame, feature_cols: list[str]
) -> dict[str, Any]:
    numeric = frame[feature_cols].apply(pd.to_numeric, errors="raise")
    return {
        "dataset": name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "label_counts": frame[LABEL_COLUMN].value_counts().sort_index().to_dict(),
        "duplicate_variant_ids": int(variant_keys(frame).duplicated().sum()),
        "duplicate_complete_rows": int(frame.duplicated().sum()),
        "missing_feature_values": int(numeric.isna().sum().sum()),
        "infinite_feature_values": int(np.isinf(numeric.to_numpy(dtype=float)).sum()),
    }


def cross_split_audit(
    train: pd.DataFrame,
    tests: dict[str, pd.DataFrame],
    feature_cols: list[str],
) -> dict[str, Any]:
    train_keys = set(variant_keys(train))
    train_feature_hash = set(
        pd.util.hash_pandas_object(train[feature_cols], index=False).astype("uint64")
    )
    report: dict[str, Any] = {}
    for name, frame in tests.items():
        keys = variant_keys(frame)
        hashes = pd.util.hash_pandas_object(
            frame[feature_cols], index=False
        ).astype("uint64")
        report[name] = {
            "variant_id_overlap_with_train": int(keys.isin(train_keys).sum()),
            "exact_feature_row_overlap_with_train": int(
                sum(item in train_feature_hash for item in hashes)
            ),
        }
    if any(
        values["variant_id_overlap_with_train"] > 0
        or values["exact_feature_row_overlap_with_train"] > 0
        for values in report.values()
    ):
        raise RuntimeError(
            "Train/test overlap detected. External-test evaluation was aborted."
        )
    return report


def calculate_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> dict[str, Any]:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "n": len(y_true),
        "threshold": threshold,
        "SEN": recall_score(y_true, predicted, zero_division=0),
        "SPE": tn / (tn + fp) if (tn + fp) else 0.0,
        "PRE": precision_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predicted),
        "ACC": accuracy_score(y_true, predicted),
        "AUC": roc_auc_score(y_true, probability),
        "AUPR": average_precision_score(y_true, probability),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def bootstrap_auc_intervals(
    y_true: np.ndarray,
    probability: np.ndarray,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    auc_values: list[float] = []
    aupr_values: list[float] = []
    for _ in range(replicates):
        sample = rng.integers(0, len(y_true), size=len(y_true))
        sampled_y = y_true[sample]
        if np.unique(sampled_y).size < 2:
            continue
        sampled_probability = probability[sample]
        auc_values.append(roc_auc_score(sampled_y, sampled_probability))
        aupr_values.append(
            average_precision_score(sampled_y, sampled_probability)
        )
    return {
        "method": "nonparametric bootstrap",
        "confidence_level": 0.95,
        "requested_replicates": replicates,
        "valid_replicates": len(auc_values),
        "AUC_95CI": np.quantile(auc_values, [0.025, 0.975]).tolist(),
        "AUPR_95CI": np.quantile(aupr_values, [0.025, 0.975]).tolist(),
    }


def predict_external(
    frame: pd.DataFrame,
    feature_cols: list[str],
    variance_filter: VarianceThreshold,
    preprocessor: Pipeline,
    models: dict[str, BaseEstimator],
    feature_indices: dict[str, np.ndarray],
    meta_model: LogisticRegression,
) -> tuple[pd.DataFrame, np.ndarray]:
    X_raw = frame[feature_cols].to_numpy(dtype=float)
    X_processed = preprocessor.transform(variance_filter.transform(X_raw))
    base_probability = np.column_stack(
        [
            models[family].predict_proba(
                X_processed[:, feature_indices[family]]
            )[:, 1]
            for family in MODEL_ORDER
        ]
    )
    final_probability = meta_model.predict_proba(base_probability)[:, 1]
    output = frame[ID_COLUMNS + [LABEL_COLUMN]].copy()
    output["probability_strict_oof_stack"] = final_probability
    output["predicted_label_threshold_0.5"] = (
        final_probability >= CLASSIFICATION_THRESHOLD
    ).astype(int)
    for model_index, family in enumerate(MODEL_ORDER):
        output[f"probability_{family}"] = base_probability[:, model_index]
    return output, final_probability


def validate_schema(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    labels = set(pd.unique(frame[LABEL_COLUMN].dropna()))
    if not labels.issubset({0, 1}):
        raise ValueError(f"{name} contains non-binary labels: {sorted(labels)}")


def environment_manifest() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
    }


def main() -> None:
    started = time.time()
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {run_dir}", flush=True)

    # Phase 1: development data only. External tests are deliberately unopened.
    train_df = pd.read_csv(TRAIN_FILE)
    feature_cols = [
        column
        for column in train_df.columns
        if column not in EXCLUDE_COLUMNS and column not in FEATURES_TO_REMOVE
    ]
    validate_schema(
        train_df, ID_COLUMNS + [LABEL_COLUMN] + feature_cols, "train"
    )
    X = train_df[feature_cols].to_numpy(dtype=float)
    y = train_df[LABEL_COLUMN].to_numpy(dtype=int)
    if not np.isfinite(X[~np.isnan(X)]).all():
        raise ValueError("Training features contain positive or negative infinity.")

    print(
        f"Development data: {X.shape[0]} samples, {X.shape[1]} features; "
        f"labels={dict(pd.Series(y).value_counts().sort_index())}",
        flush=True,
    )
    tuning_rows: list[dict[str, Any]] = []
    (
        base_oof,
        nested_stack_probability,
        fold_assignment,
        fold_records,
    ) = strict_nested_oof(X, y, tuning_rows)

    oof_frame = train_df[ID_COLUMNS + [LABEL_COLUMN]].copy()
    oof_frame["outer_fold"] = fold_assignment
    for model_index, family in enumerate(MODEL_ORDER):
        oof_frame[f"oof_probability_{family}"] = base_oof[:, model_index]
    oof_frame["nested_cv_probability_stack"] = nested_stack_probability
    oof_frame.to_csv(
        run_dir / "training_oof_predictions.csv",
        index=False,
        float_format="%.10g",
    )
    pd.DataFrame(tuning_rows).to_csv(
        run_dir / "tuning_fold_results.csv",
        index=False,
        float_format="%.10g",
    )
    write_json(run_dir / "outer_fold_records.json", fold_records)

    nested_metrics = calculate_metrics(y, nested_stack_probability)
    nested_metrics["intervals"] = bootstrap_auc_intervals(
        y, nested_stack_probability, derive_seed(107)
    )
    write_json(run_dir / "nested_cv_metrics.json", nested_metrics)

    (
        variance_filter,
        preprocessor,
        models,
        feature_indices,
        meta_model,
        final_selected,
        final_selection_record,
    ) = fit_final_models(
        X, y, base_oof, feature_cols, tuning_rows, run_dir
    )
    # fit_final_models appends final-tuning rows.
    pd.DataFrame(tuning_rows).to_csv(
        run_dir / "tuning_fold_results.csv",
        index=False,
        float_format="%.10g",
    )

    model_lock = {
        "locked_before_external_test_load": True,
        "locked_at": datetime.now().isoformat(timespec="seconds"),
        "training_file": str(TRAIN_FILE.relative_to(PROJECT_ROOT)),
        "training_sha256": sha256_file(TRAIN_FILE),
        "n_training_samples": len(train_df),
        "n_features": len(feature_cols),
        "random_seed": RANDOM_SEED,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "final_tuning_splits": FINAL_TUNING_SPLITS,
        "repeats": REPEATS,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "model_order": MODEL_ORDER,
        "final_selection": final_selection_record,
    }
    write_json(run_dir / "MODEL_LOCK_BEFORE_TEST.json", model_lock)
    print(
        "\n[LOCKED] Training and model selection complete. "
        "External tests will now be opened once.",
        flush=True,
    )

    # Phase 2: one-time external test prediction and evaluation.
    test_frames = {
        name: pd.read_csv(path) for name, path in TEST_FILES.items()
    }
    required_columns = ID_COLUMNS + [LABEL_COLUMN] + feature_cols
    for name, frame in test_frames.items():
        validate_schema(frame, required_columns, name)

    audit = {
        "train": dataset_summary("train", train_df, feature_cols),
        **{
            name: dataset_summary(name, frame, feature_cols)
            for name, frame in test_frames.items()
        },
        "cross_split": cross_split_audit(train_df, test_frames, feature_cols),
    }
    write_json(run_dir / "data_audit.json", audit)

    external_metrics: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = [
        {"dataset": "nested_training_cv", **nested_metrics}
    ]
    for test_index, (name, frame) in enumerate(test_frames.items(), start=1):
        prediction_frame, probability = predict_external(
            frame,
            feature_cols,
            variance_filter,
            preprocessor,
            models,
            feature_indices,
            meta_model,
        )
        prediction_path = run_dir / f"{name}_strict_oof_predictions.csv"
        prediction_frame.to_csv(
            prediction_path, index=False, float_format="%.10g"
        )
        metrics = calculate_metrics(
            frame[LABEL_COLUMN].to_numpy(dtype=int), probability
        )
        metrics["intervals"] = bootstrap_auc_intervals(
            frame[LABEL_COLUMN].to_numpy(dtype=int),
            probability,
            derive_seed(test_index, 109),
        )
        external_metrics[name] = metrics
        metric_rows.append({"dataset": name, **metrics})
        print(
            f"{name}: AUC={metrics['AUC']:.4f}, "
            f"AUPR={metrics['AUPR']:.4f}, "
            f"ACC={metrics['ACC']:.4f}, MCC={metrics['MCC']:.4f}",
            flush=True,
        )
    write_json(run_dir / "external_test_metrics.json", external_metrics)

    flat_metric_rows = []
    for row in metric_rows:
        flat_metric_rows.append(
            {
                key: value
                for key, value in row.items()
                if key != "intervals"
            }
        )
    pd.DataFrame(flat_metric_rows).to_csv(
        run_dir / "metrics_summary.csv", index=False, float_format="%.10g"
    )
    requested_metric_columns = [
        "Dataset",
        "SEN",
        "SPE",
        "PRE",
        "F1",
        "MCC",
        "ACC",
        "AUC",
        "AUPR",
    ]
    requested_metrics = pd.DataFrame(flat_metric_rows).rename(
        columns={"dataset": "Dataset"}
    )[requested_metric_columns]
    requested_metrics.to_csv(
        run_dir / "all_requested_metrics.csv",
        index=False,
        float_format="%.10g",
    )

    manifest = {
        "run_name": run_name,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.time() - started,
        "status": "complete",
        "design": {
            "stacking": "strict nested OOF",
            "outer_cv": f"{OUTER_SPLITS}-fold StratifiedKFold",
            "inner_tuning_cv": f"{INNER_SPLITS}-fold StratifiedKFold",
            "final_tuning_cv": f"{FINAL_TUNING_SPLITS}-fold StratifiedKFold",
            "repeats": REPEATS,
            "hyperparameter_selection_metric": "ROC AUC; AUPR tie-break",
            "meta_learner": "L2 LogisticRegression",
            "threshold": CLASSIFICATION_THRESHOLD,
            "test_access": "after model lock only; one prediction/evaluation pass",
        },
        "inputs": {
            "train": {
                "path": str(TRAIN_FILE.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(TRAIN_FILE),
            },
            **{
                name: {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_file(path),
                }
                for name, path in TEST_FILES.items()
            },
        },
        "environment": environment_manifest(),
        "outputs": sorted(
            [
                path.relative_to(PROJECT_ROOT).as_posix()
                for output_dir in [run_dir, MODEL_DIR]
                for path in output_dir.rglob("*")
                if path.is_file()
            ]
        ),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    print(
        f"\nComplete in {manifest['elapsed_seconds'] / 60:.1f} minutes.\n"
        f"All outputs: {run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    # Prevent accidental oversubscription differences caused by implicit BLAS
    # settings while keeping tree-library parallelism available.
    os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_SEED))
    main()
