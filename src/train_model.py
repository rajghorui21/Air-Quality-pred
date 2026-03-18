import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os
from data_utils import load_and_clean_data, create_sliding_windows

def train_aqi_model(data_path="data/air_quality.csv", model_dir="models"):
    """
    Train XGBoost model on sliding window data aqi prediction.
    """
    print("Loading data...")
    df = load_and_clean_data(data_path)
    
    window_size = 24
    target_col = 'AQI'
    print(f"Creating sliding windows (Window Size: {window_size} hours)...")
    X, y = create_sliding_windows(df, window_size=window_size, target_col=target_col)
    
    # Flatten X from (N, L, F) to (N, L * F) for XGBoost
    N, L, F = X.shape
    X_flat = X.reshape(N, L * F)
    
    print(f"Flattened X shape: {X_flat.shape}")
    
    # Time-Series Split: Chronological (No Shuffle)
    # Split index (80% Train, 20% Test)
    split_idx = int(N * 0.8)
    X_train, X_test = X_flat[:split_idx], X_flat[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train Model
    print("Training XGBoost Regressor...")
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predict & Evaluate
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Save Model Weights
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "aqi_xgb_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature names/metadata for pipeline use
    metadata = {
        'window_size': window_size,
        'features': list(df.columns),
        'target_col': target_col,
        'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2}
    }
    joblib.dump(metadata, os.path.join(model_dir, "model_metadata.joblib"))
    print("Metadata saved.")

if __name__ == "__main__":
    train_aqi_model()
