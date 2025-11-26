# -*- coding: utf-8 -*-
import sys
import json
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Windows encoding fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from scripts.ml_optimization.run_ml_pipeline import MLPipeline
from app.services.ml.label_cache import LabelCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_run(run_dir_name: str):
    # Initialize pipeline (will create a new run dir, but we ignore it)
    pipeline = MLPipeline()

    # Path to the results we want to evaluate
    results_dir = (
        Path(__file__).parent.parent.parent / "results" / "ml_pipeline" / run_dir_name
    )

    if not results_dir.exists():
        logger.error(f"Directory not found: {results_dir}")
        return

    logger.info(f"Evaluating run: {run_dir_name}")

    # Load params
    try:
        with open(results_dir / "best_params.json", "r", encoding="utf-8") as f:
            params = json.load(f)
            best_params = params["best_params"]
    except Exception as e:
        logger.error(f"Error loading params: {e}")
        return

    # Load models
    models = {}
    try:
        if (results_dir / "model_lgb.joblib").exists():
            models["lightgbm"] = joblib.load(results_dir / "model_lgb.joblib")
        if (results_dir / "model_xgb.joblib").exists():
            models["xgboost"] = joblib.load(results_dir / "model_xgb.joblib")
        if (results_dir / "model_cat.joblib").exists():
            models["catboost"] = joblib.load(results_dir / "model_cat.joblib")

        stacking_service = joblib.load(results_dir / "stacking_service.joblib")
        meta_service = joblib.load(results_dir / "meta_labeling_service.joblib")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return

    # Prepare data
    # We use the pipeline's method to fetch data
    X, ohlcv = pipeline.prepare_data(limit=20000)

    # Recreate labels using the loaded params
    logger.info("Recreating labels...")
    label_cache = LabelCache(ohlcv)

    # Find the threshold key (e.g., quantile_threshold, volatility_threshold)
    threshold_key = [
        k for k in best_params if "threshold" in k and k != "threshold_method"
    ][0]

    labels = label_cache.get_labels(
        horizon_n=best_params["horizon_n"],
        threshold_method=best_params["threshold_method"],
        threshold=best_params[threshold_key],
        timeframe=pipeline.timeframe,
        price_column="close",
    )

    # Align data
    common_index = X.index.intersection(labels.index)
    X_aligned = X.loc[common_index]
    labels_aligned = labels.loc[common_index]
    valid_idx = ~labels_aligned.isna()
    X_clean = X_aligned.loc[valid_idx]
    y = labels_aligned.loc[valid_idx].map({"DOWN": 1, "RANGE": 0, "UP": 1})

    # Check index monotonicity
    if not X_clean.index.is_monotonic_increasing:
        logger.error("X_clean index is NOT monotonic increasing!")
    else:
        logger.info("X_clean index is monotonic increasing.")

    # OOS Split (Chronological 80/20)
    split_idx = int(len(X_clean) * 0.8)
    X_train_oos = X_clean.iloc[:split_idx]
    X_test_oos = X_clean.iloc[split_idx:]
    y_train_oos = y.iloc[:split_idx]
    y_test_oos = y.iloc[split_idx:]

    logger.info(f"Train range: {X_train_oos.index.min()} to {X_train_oos.index.max()}")
    logger.info(f"Test range: {X_test_oos.index.min()} to {X_test_oos.index.max()}")

    if X_train_oos.index.max() >= X_test_oos.index.min():
        logger.error("Train and Test sets OVERLAP!")

    logger.info(f"OOS Test Set: {len(X_test_oos)} samples")
    logger.info(f"Class Distribution in Test Set:\n{y_test_oos.value_counts()}")

    # Predict with base models
    oos_preds = {}
    if "lightgbm" in models:
        oos_preds["LightGBM"] = models["lightgbm"].predict_proba(X_test_oos)[:, 1]
    if "xgboost" in models:
        oos_preds["XGBoost"] = models["xgboost"].predict_proba(X_test_oos)[:, 1]
    if "catboost" in models:
        oos_preds["CatBoost"] = models["catboost"].predict_proba(X_test_oos)[:, 1]

    oos_preds_df = pd.DataFrame(oos_preds, index=X_test_oos.index)

    # Stacking Prediction
    logger.info("Running Stacking Prediction...")
    stack_preds = stacking_service.predict(oos_preds_df)

    # Meta Labeling Evaluation
    logger.info("Evaluating Meta-Labeling...")
    primary_test_series = pd.Series(stack_preds, index=X_test_oos.index)

    meta_eval = meta_service.evaluate(
        X_test=X_test_oos,
        y_test=y_test_oos,
        primary_proba_test=primary_test_series,
        base_model_probs_df=oos_preds_df,
    )

    output_file = results_dir / "oos_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== OOS Evaluation Results ===\n")
        f.write(f"Test Set Size: {len(y_test_oos)} samples\n")
        f.write(
            f"Primary Model Precision (OOS): {meta_eval.get('primary_precision', 0):.1%}\n"
        )
        f.write(f"Meta Model Precision (OOS): {meta_eval['meta_precision']:.1%}\n")
        f.write(f"Meta Model Recall (OOS): {meta_eval['meta_recall']:.1%}\n")
        f.write(f"Meta Model Accuracy (OOS): {meta_eval['meta_accuracy']:.1%}\n")
        f.write(
            f"Precision Improvement (OOS): {meta_eval['improvement_precision']:.1%}\n"
        )

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Use the run ID found in the previous step
    evaluate_run("run_20251126_091757")
