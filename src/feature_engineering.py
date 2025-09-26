# 02_feature_engineering.py

import pandas as pd
import numpy as np

def create_features(df):
    """
    Add features required for modeling:
    - Normalize sensors
    - Add RUL if missing
    - Add rolling statistics (mean, std)
    - Add differences (1st, 2nd order)
    - Add cumulative stats (sum, max, min)
    - Add cycle-based normalized feature
    """
    engine_col, cycle_col = df.columns[0], df.columns[1]
    sensor_cols = df.columns[2:]

    # -----------------------------
    # 1. Normalize sensors (z-score)
    # -----------------------------
    df_norm = (df[sensor_cols] - df[sensor_cols].mean()) / df[sensor_cols].std(ddof=0)

    # -----------------------------
    # 2. Add Remaining Useful Life (RUL)
    # -----------------------------
    if "RUL" not in df.columns:
        df_norm["RUL"] = df.groupby(engine_col)[cycle_col].transform(lambda x: x.max() - x)
    else:
        df_norm["RUL"] = df["RUL"]

    # -----------------------------
    # 3. Temporal / rolling features
    # -----------------------------
    feature_list = []
    for col in sensor_cols:
        grouped = df.groupby(engine_col)[col]
        features = [
            grouped.rolling(5, min_periods=1).mean().reset_index(0, drop=True).rename(f"{col}_roll_mean"),
            grouped.rolling(5, min_periods=1).std().reset_index(0, drop=True).fillna(0).rename(f"{col}_roll_std"),
            grouped.diff().fillna(0).rename(f"{col}_diff1"),
            grouped.diff().diff().fillna(0).rename(f"{col}_diff2"),
            grouped.cumsum().rename(f"{col}_cumsum"),
            grouped.cummax().rename(f"{col}_cummax"),
            grouped.cummin().rename(f"{col}_cummin")
        ]
        feature_list.extend(features)

    # -----------------------------
    # 4. Cycle-based normalized feature
    # -----------------------------
    cycle_norm = df.groupby(engine_col)[cycle_col].transform(lambda x: x / x.max()).rename("cycle_norm")

    # Concatenate all features at once
    df_features = pd.concat([df_norm] + feature_list + [cycle_norm], axis=1)

    # Defragment DataFrame for memory efficiency
    df_features = df_features.copy()

    return df_features


if __name__ == "__main__":
    from data_exploration import load_data

    raw_file = "data/raw/train_FD001.txt"
    df = load_data(raw_file)
    df_feat = create_features(df)
    print("✅ Features added. Head:\n", df_feat.head())
    print(f"📊 Total columns after feature engineering: {df_feat.shape[1]}")
