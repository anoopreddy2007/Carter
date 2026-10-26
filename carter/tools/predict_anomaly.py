"""
Tool: predict_anomaly
Loads the saved ML model artifact and runs inference on validated feature input.
All inference is local — no remote calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carter.utils.logging import get_logger

logger = get_logger("tools.predict_anomaly")


def predict_anomaly(
    features: list[float],
    model_path: str,
    model_type: str = "sklearn",
    max_features: int = 50,
) -> dict[str, Any]:
    """
    Run the local ML model on a feature vector and return a prediction.

    Args:
        features: A list of numeric feature values.
        model_path: Local path to the saved model artifact.
        model_type: "sklearn" or "pytorch".
        max_features: Maximum allowed feature count.

    Returns:
        A dict with:
          - "prediction": the model's output class/value
          - "confidence": confidence score if available
          - "model_type": the model type used

    Raises:
        ValueError: on invalid input.
        FileNotFoundError: if the model artifact is missing.
        ImportError: if required packages are not installed.
    """
    # Input validation
    if not isinstance(features, list):
        raise ValueError("features must be a list of numbers.")

    if len(features) == 0:
        raise ValueError("features list is empty.")

    if len(features) > max_features:
        raise ValueError(
            f"Too many features: {len(features)}. Maximum allowed: {max_features}."
        )

    for i, v in enumerate(features):
        if not isinstance(v, (int, float)):
            raise ValueError(
                f"Feature at index {i} is not a number: {v!r}"
            )

    # Model file check
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ML model artifact not found: {model_path}\n"
            "Train the model first: python -m carter.ml.training.train"
        )

    if model_type == "sklearn":
        return _predict_sklearn(features, path)
    elif model_type == "pytorch":
        return _predict_pytorch(features, path)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Use 'sklearn' or 'pytorch'.")


def _predict_sklearn(features: list[float], model_path: Path) -> dict[str, Any]:
    """Run inference with a scikit-learn model loaded via joblib."""
    try:
        import joblib
        import numpy as np
    except ImportError as e:
        raise ImportError(
            f"Required package not installed: {e}. "
            "Install from the local wheelhouse: pip install scikit-learn joblib numpy"
        ) from e

    model = joblib.load(model_path)
    X = np.array(features, dtype=float).reshape(1, -1)

    prediction = model.predict(X)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
            confidence = float(max(proba))
        except Exception:
            pass
    elif hasattr(model, "decision_function"):
        try:
            decision = model.decision_function(X)[0]
            confidence = float(decision)
        except Exception:
            pass

    logger.info(
        "predict_anomaly (sklearn): prediction=%s, confidence=%s",
        prediction, confidence
    )

    result: dict[str, Any] = {
        "prediction": int(prediction) if hasattr(prediction, "item") else prediction,
        "model_type": "sklearn",
    }
    if confidence is not None:
        result["confidence"] = round(confidence, 4)

    return result


def _predict_pytorch(features: list[float], model_path: Path) -> dict[str, Any]:
    """Run inference with a PyTorch model loaded from a checkpoint."""
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            f"PyTorch not installed: {e}. "
            "Install from the local wheelhouse."
        ) from e

    # NOTE: Placeholder — actual architecture must match training code.
    # The training module will save both weights and architecture config.
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    raise NotImplementedError(
        "PyTorch inference requires the model architecture to be implemented "
        "in carter.ml.inference. This stub will be completed in Week 4."
    )
