# scripts/predict.py

import os
import pickle
import pandas as pd
from src.feature_engineering import create_features
from src.data_exploration import load_data


def load_model(model_name=None):
    """Load a trained model. Defaults to best_model.pkl if no model_name is provided."""
    model_path = os.path.join('models', f'{model_name}_model.pkl') if model_name else 'models/best_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{os.path.basename(model_path)} not found. Train models first using run_pipeline.py.")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Loaded model from {model_path}")
    return model


def load_feature_columns():
    """Load feature columns used during training."""
    feature_columns_path = 'models/feature_columns.pkl'
    if not os.path.exists(feature_columns_path):
        raise FileNotFoundError("feature_columns.pkl not found. Run feature engineering first.")
    with open(feature_columns_path, 'rb') as f:
        return pickle.load(f)


def load_imputer():
    """Load imputer fitted during training."""
    imputer_path = 'models/imputer.pkl'
    if not os.path.exists(imputer_path):
        raise FileNotFoundError("imputer.pkl not found. Run data preparation first.")
    with open(imputer_path, 'rb') as f:
        return pickle.load(f)


def predict(file_path, model_name=None, last_cycle_only=False):
    """
    Predict RUL for the given dataset using the trained model.

    Args:
        file_path (str): Path to test data
        model_name (str): Name of the trained model to use
        last_cycle_only (bool): If True, return predictions only for last cycle per engine
    """
    # Load and preprocess data
    df = load_data(file_path)
    df = create_features(df)
    df = df.dropna(axis=1, how='all')

    if df.shape[1] < 2:
        raise ValueError("Expected at least two columns (engine_id and cycle). Check input data.")

    # Detect engine and cycle columns
    engine_col, cycle_col = df.columns[0], df.columns[1]
    df[engine_col] = df[engine_col].astype(int)

    # Align features with training
    feature_columns = load_feature_columns()
    aligned_columns = [col for col in feature_columns if col in df.columns]

    if not aligned_columns:
        raise ValueError("None of the feature columns from training are present in the test set.")

    X = df[aligned_columns].values

    # Apply saved imputer
    imputer = load_imputer()
    X = imputer.transform(X)

    # Predict
    model = load_model(model_name)
    df['Predicted_RUL'] = model.predict(X)

    # Keep only last cycle per engine if requested
    if last_cycle_only:
        if cycle_col not in df.columns:
            raise ValueError(f"Cycle column '{cycle_col}' not found in the dataset.")
        df = df.sort_values(by=[engine_col, cycle_col])
        df = df.groupby(engine_col, as_index=False).last()
        print(f"✅ Returning predictions for last cycle only ({len(df)} engines)")

    return df


if __name__ == "__main__":
    test_file = 'data/raw/test_FD001.txt'
    try:
        predictions = predict(test_file, last_cycle_only=True)
    except Exception as e:
        print(f"\n⚠️ Prediction failed: {e}")
        exit(1)

    output_file = 'data/processed/predictions_FD001.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    predictions.to_csv(output_file, index=False)
    print(f"\n✅ Predictions saved to {output_file}")
    print(predictions.head())
