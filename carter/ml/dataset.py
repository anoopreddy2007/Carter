"""
Dataset generation and preprocessing for offline ML anomaly detection.
Generates synthetic system telemetry (CPU, Memory, Disk I/O, Network, Latency).
Deterministic and reproducible via random seed.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Tuple
import numpy as np


FEATURE_NAMES = [
    "cpu_percent",
    "memory_percent",
    "disk_io_mbs",
    "network_kbs",
    "request_latency_ms",
]


def generate_synthetic_telemetry(
    num_samples: int = 1000,
    anomaly_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic host/service telemetry metrics.

    Normal patterns:
    - cpu_percent: normal distribution ~ 35 +/- 12 (clipped 5..95)
    - memory_percent: normal distribution ~ 45 +/- 10 (clipped 10..95)
    - disk_io_mbs: normal distribution ~ 15 +/- 6 (clipped 1..80)
    - network_kbs: normal distribution ~ 120 +/- 35 (clipped 10..400)
    - request_latency_ms: lognormal distribution ~ 25 +/- 8 (clipped 5..150)

    Anomalous patterns (label = -1):
    - Spikes in CPU (> 90%), memory saturation, runaway latency (> 500ms), or disk thrashing.

    Returns:
        X: np.ndarray of shape (num_samples, 5)
        y: np.ndarray of shape (num_samples,), where 1 = normal, -1 = anomaly
    """
    rng = np.random.RandomState(seed)

    num_anomalies = int(num_samples * anomaly_ratio)
    num_normals = num_samples - num_anomalies

    # 1. Normal samples
    cpu_norm = rng.normal(loc=35.0, scale=10.0, size=num_normals).clip(5.0, 75.0)
    mem_norm = rng.normal(loc=45.0, scale=8.0, size=num_normals).clip(15.0, 75.0)
    disk_norm = rng.normal(loc=15.0, scale=5.0, size=num_normals).clip(1.0, 50.0)
    net_norm = rng.normal(loc=120.0, scale=30.0, size=num_normals).clip(10.0, 300.0)
    lat_norm = rng.normal(loc=25.0, scale=6.0, size=num_normals).clip(5.0, 60.0)

    X_normal = np.column_stack([cpu_norm, mem_norm, disk_norm, net_norm, lat_norm])
    y_normal = np.ones(num_normals, dtype=int)

    # 2. Anomaly samples (outliers across different failure modes)
    X_anom = np.empty((num_anomalies, 5), dtype=float)
    for i in range(num_anomalies):
        mode = rng.choice(["cpu_spike", "mem_leak", "latency_freeze", "disk_thrash"])
        if mode == "cpu_spike":
            X_anom[i] = [
                rng.uniform(92.0, 100.0),
                rng.uniform(50.0, 80.0),
                rng.uniform(20.0, 60.0),
                rng.uniform(200.0, 500.0),
                rng.uniform(80.0, 200.0),
            ]
        elif mode == "mem_leak":
            X_anom[i] = [
                rng.uniform(40.0, 70.0),
                rng.uniform(95.0, 99.5),
                rng.uniform(10.0, 40.0),
                rng.uniform(50.0, 150.0),
                rng.uniform(100.0, 300.0),
            ]
        elif mode == "latency_freeze":
            X_anom[i] = [
                rng.uniform(20.0, 60.0),
                rng.uniform(40.0, 70.0),
                rng.uniform(5.0, 25.0),
                rng.uniform(5.0, 40.0),
                rng.uniform(450.0, 1500.0),
            ]
        else:  # disk_thrash
            X_anom[i] = [
                rng.uniform(70.0, 95.0),
                rng.uniform(60.0, 85.0),
                rng.uniform(180.0, 350.0),
                rng.uniform(50.0, 200.0),
                rng.uniform(120.0, 400.0),
            ]

    y_anom = -np.ones(num_anomalies, dtype=int)

    X = np.vstack([X_normal, X_anom])
    y = np.concatenate([y_normal, y_anom])

    # Shuffle synchronously
    indices = rng.permutation(num_samples)
    return X[indices], y[indices]


def save_dataset_csv(
    X: np.ndarray,
    y: np.ndarray,
    output_path: Path | str,
) -> dict[str, Any]:
    """
    Save feature matrix and labels to CSV and compute a deterministic SHA256 hash.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES + ["label"])
        for row, label in zip(X, y):
            writer.writerow([round(v, 4) for v in row] + [int(label)])

    # Compute sha256
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)

    return {
        "path": str(path),
        "num_rows": len(y),
        "num_features": X.shape[1],
        "feature_names": FEATURE_NAMES,
        "sha256": hasher.hexdigest(),
    }


def load_dataset_csv(filepath: Path | str) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Load dataset from CSV file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        feat_names = header[:-1]
        for line in reader:
            if not line:
                continue
            rows.append([float(val) for val in line[:-1]])
            labels.append(int(line[-1]))

    return np.array(rows, dtype=float), np.array(labels, dtype=int), feat_names

