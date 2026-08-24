"""Run inference with a trained strict-OOF IGRM model package."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model_pkl"
DEFAULT_THRESHOLD = 0.5
ID_COLUMNS = ["chr", "pos", "ref", "alt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict positive-class probabilities with the IGRM stack."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV containing all required model features.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV for probabilities and predicted labels.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Classification threshold (default: 0.5).",
    )
    return parser.parse_args()


def load_model_package(model_dir: Path) -> dict[str, object]:
    required = [
        "variance_filter.pkl",
        "preprocessor.pkl",
        "feature_cols.pkl",
        "feature_indices.pkl",
        "model_names.pkl",
        "LR_meta_oof.pkl",
    ]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Model package is incomplete; missing files: {missing}"
        )

    model_names = joblib.load(model_dir / "model_names.pkl")
    models = {}
    for name in model_names:
        model_path = model_dir / f"{name}_base.pkl"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing base model: {model_path}")
        models[name] = joblib.load(model_path)

    return {
        "variance_filter": joblib.load(model_dir / "variance_filter.pkl"),
        "preprocessor": joblib.load(model_dir / "preprocessor.pkl"),
        "feature_cols": joblib.load(model_dir / "feature_cols.pkl"),
        "feature_indices": joblib.load(model_dir / "feature_indices.pkl"),
        "model_names": model_names,
        "meta_model": joblib.load(model_dir / "LR_meta_oof.pkl"),
        "models": models,
    }


def predict(
    frame: pd.DataFrame,
    package: dict[str, object],
    threshold: float,
) -> pd.DataFrame:
    feature_cols = package["feature_cols"]
    missing = sorted(set(feature_cols) - set(frame.columns))
    if missing:
        raise ValueError(
            f"Input CSV is missing {len(missing)} required features: "
            f"{missing[:20]}"
        )

    X_raw = frame[feature_cols].apply(pd.to_numeric, errors="raise").to_numpy()
    X_processed = package["preprocessor"].transform(
        package["variance_filter"].transform(X_raw)
    )
    model_names = package["model_names"]
    feature_indices = package["feature_indices"]
    models = package["models"]
    base_probability = np.column_stack(
        [
            models[name].predict_proba(
                X_processed[:, feature_indices[name]]
            )[:, 1]
            for name in model_names
        ]
    )
    final_probability = package["meta_model"].predict_proba(
        base_probability
    )[:, 1]

    metadata = [column for column in ID_COLUMNS if column in frame.columns]
    if "Label" in frame.columns:
        metadata.append("Label")
    output = frame[metadata].copy() if metadata else pd.DataFrame(index=frame.index)
    output["probability_strict_oof_stack"] = final_probability
    output["predicted_label"] = (final_probability >= threshold).astype(int)
    for index, name in enumerate(model_names):
        output[f"probability_{name}"] = base_probability[:, index]
    return output


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    package = load_model_package(args.model_dir)
    frame = pd.read_csv(args.input)
    output = predict(frame, package, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, float_format="%.10g")
    print(f"Predicted {len(output)} samples.")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
