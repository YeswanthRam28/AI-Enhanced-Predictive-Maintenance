# model_training.py

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

from feature_engineering import create_features
from data_exploration import load_data


def prepare_data(file_path):
    """
    Load raw data, apply feature engineering, handle missing values,
    and split into train/test sets.
    """
    # Load and preprocess data
    df = load_data(file_path)
    df = create_features(df)

    # Drop columns with all missing values
    df = df.dropna(axis=1, how='all')

    # Select feature columns (exclude ID, cycle, RUL)
    feature_columns = df.drop(columns=[df.columns[0], df.columns[1], 'RUL']).columns.tolist()

    # Save feature columns
    os.makedirs('models', exist_ok=True)
    feature_columns_path = os.path.join('models', 'feature_columns.pkl')
    with open(feature_columns_path, 'wb') as f:
        pickle.dump(feature_columns, f)
    print(f"Feature columns saved at {feature_columns_path}")

    # Prepare features and target
    X = df[feature_columns].values
    y = df['RUL'].values

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Save imputer
    imputer_path = os.path.join('models', 'imputer.pkl')
    with open(imputer_path, 'wb') as f:
        pickle.dump(imputer, f)
    print(f"Imputer saved at {imputer_path}")

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple regression models, evaluate them,
    save each model, and return the best model and results.
    """
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }

    results = {}
    trained_models = {}
    os.makedirs('models', exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {"RMSE": rmse, "R2": r2}
        trained_models[name] = model

        # Save each trained model
        model_path = os.path.join('models', f"{name.lower()}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"{name} trained and saved at {model_path}")

    # Select the best model (lowest RMSE)
    best_model_name = min(results, key=lambda k: results[k]["RMSE"])
    best_model = trained_models[best_model_name]

    best_model_path = os.path.join('models', "best_model.pkl")
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n✅ Best model: {best_model_name} saved at {best_model_path}")

    return best_model_name, results


if __name__ == "__main__":
    raw_file = 'data/raw/train_FD001.txt'
    X_train, X_test, y_train, y_test = prepare_data(raw_file)

    results, best_model_name = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    print("\nModel Comparison:")
    for name, metrics in results.items():
        print(f"{name}: RMSE={metrics['RMSE']:.3f}, R²={metrics['R2']:.3f}")

    print(f"\n✅ Best Model Selected: {best_model_name}")
