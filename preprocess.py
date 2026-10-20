"""
preprocess.py
=============
Template for the ML Assignment preprocessing pipeline.

Instructions
------------
1. Implement your preprocessing steps in the `preprocess()` function.
2. Do NOT change the input/output filenames or the overall structure.


       python preprocess.py

   It will read  X_test.csv  and write  X_test_preprocessed.csv
   in the same directory.

Notes
-----
- Any parameters you fitted on the training set (e.g. mean, std, min, max)
  must be hard-coded here — do not load external files.
- The output must have the same number of rows as the input.
"""

import pandas as pd
import numpy as np

# =============================================================================
# PREPROCESSING FUNCTION — implement your pipeline here
# =============================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps to the raw feature DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw features loaded from X_test.csv.

    Returns
    -------
    pd.DataFrame
        Preprocessed features, ready for model inference.
    """

    df = df.copy()

    # Expected feature order from X_train.csv
    feature_cols = ["x1", "x2", "x3", "x4"]

    # Training-set medians for NaN imputation
    medians = {
        "x1": -0.006345,
        "x2": 29.723467,
        "x3": 2472.062064,
        "x4": 0.511433,
    }

    # 1st and 99th percentiles from training set for winsorization
    clip_bounds = {
        "x1": (-1.356874, 1.354135),
        "x2": (-1372.417968, 1371.640287),
        "x3": (72.854715, 4972.143945),
        "x4": (0.010989, 0.991867),
    }

    # Training-set mean/std for z-score scaling
    means = {
        "x1": -0.006507,
        "x2": 18.571989,
        "x3": 2501.138903,
        "x4": 0.505283,
    }
    stds = { #standard deviation
        "x1": 0.746068,
        "x2": 747.720234,
        "x3": 1451.415773,
        "x4": 0.285221,
    }

    # Normalize incoming headers and safely rebuild expected feature set.
    df.columns = [str(c).strip() for c in df.columns]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Keep only expected model features in training order.
    df = df[feature_cols]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(medians[col])

        lo, hi = clip_bounds[col]
        df[col] = df[col].clip(lower=lo, upper=hi)

        df[col] = (df[col] - means[col]) / stds[col]

    return df

if __name__ == "__main__":
    INPUT_FILE  = "X_train.csv"
    OUTPUT_FILE = "X_test_preprocessed.csv"

    print(f"Reading {INPUT_FILE} ...")
    raw = pd.read_csv(INPUT_FILE)
    print(f"  Input shape : {raw.shape}")

    processed = preprocess(raw)
    print(f"  Output shape: {processed.shape}")

    processed.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}.")
