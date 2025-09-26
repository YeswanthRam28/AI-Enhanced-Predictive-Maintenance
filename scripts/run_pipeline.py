# scripts/run_pipeline.py

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Add src folder to sys.path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model_training import prepare_data, train_and_evaluate_models
from scripts.predict import predict


def evaluate_predictions(predictions, rul_file):
    """
    Evaluate predictions against ground truth RUL.
    Handles:
    - Column mismatches
    - Subset of engines
    - Auto-renamed columns
    """
    # Load true RULs
    try:
        true_rul_df = pd.read_csv(rul_file, header=None, names=["RUL"])
    except Exception as e:
        raise RuntimeError(f"Failed to read RUL file: {rul_file}\n{e}")

    true_rul_df["Engine_ID"] = range(1, len(true_rul_df) + 1)

    # Detect engine and cycle columns
    engine_col, cycle_col = predictions.columns[0], predictions.columns[1]
    predictions[engine_col] = predictions[engine_col].astype(int)

    # Keep only last cycle per engine
    if cycle_col in predictions.columns:
        last_cycles = predictions.sort_values(by=[engine_col, cycle_col]) \
                                 .groupby(engine_col, as_index=False).last()
    else:
        last_cycles = predictions.groupby(engine_col, as_index=False).last()

    # Merge predictions with true RULs
    merged = pd.merge(
        last_cycles, true_rul_df,
        left_on=engine_col, right_on="Engine_ID",
        how="inner"
    )

    # Safely detect RUL column
    if 'RUL' not in merged.columns:
        rul_candidates = [c for c in merged.columns if 'RUL' in c and c != 'Predicted_RUL']
        if rul_candidates:
            merged.rename(columns={rul_candidates[0]: 'RUL'}, inplace=True)
        else:
            raise KeyError("No 'RUL' column found after merge. Check RUL file and predictions.")

    pred_rul = merged["Predicted_RUL"].values
    true_rul = merged["RUL"].values

    print(f"✅ Evaluating {len(pred_rul)} engines (subset if not full test set)")

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(true_rul, pred_rul))
    r2 = r2_score(true_rul, pred_rul)
    print(f"\nEvaluation on test set (last cycle per engine): RMSE={rmse:.3f}, R²={r2:.3f}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(true_rul)), true_rul, label="Actual RUL", color="blue")
    plt.scatter(range(len(pred_rul)), pred_rul, label="Predicted RUL", color="red", alpha=0.6)
    plt.xlabel("Engine Index")
    plt.ylabel("RUL")
    plt.title("Predicted vs Actual RUL (last cycle per engine)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ================================
    # Step 1: Prepare data and train models
    # ================================
    raw_file = "data/raw/train_FD001.txt"
    X_train, X_test, y_train, y_test = prepare_data(raw_file)

    best_model_name, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    print("\nTraining completed. Model comparison:")
    for name, metrics in results.items():
        print(f"{name}: RMSE={metrics['RMSE']:.3f}, R²={metrics['R2']:.3f}")

    print(f"\n✅ Best Model Selected: {best_model_name}")

    # ================================
    # Step 2: Predict on test set
    # ================================
    test_file = "data/raw/test_FD001.txt"
    try:
        predictions = predict(
            test_file,
            model_name=best_model_name.lower(),
            last_cycle_only=True
        )
    except Exception as e:
        print(f"\n⚠️ Prediction failed: {e}")
        exit(1)

    # Save predictions
    output_file = "data/processed/predictions_FD001.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    predictions.to_csv(output_file, index=False)
    print(f"\n✅ Predictions saved to {output_file}")
    print("\nSample predictions:")
    print(predictions.head())

    # ================================
    # Step 3: Evaluate predictions
    # ================================
    rul_file = "data/raw/RUL_FD001.txt"
    if os.path.exists(rul_file):
        try:
            evaluate_predictions(predictions, rul_file)
        except Exception as e:
            print(f"\n⚠️ Evaluation failed: {e}")
    else:
        print("\nGround truth RUL not found. Skipping evaluation.")
