import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_data(filepath="data/air_quality.csv", hours=8760):
    """
    Generate synthetic time-series air quality data.
    8760 hours = 1 year.
    """
    print(f"Generating synthetic data: {hours} hours...")
    np.random.seed(42)
    
    # Create time index
    start_date = datetime(2025, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
    
    # Base seasonal trends (Sine waves)
    # Day/Night cycle (24 hours)
    day_cycle = np.sin(np.linspace(0, 2 * np.pi * (hours / 24), hours))
    # Yearly cycle (8760 hours)
    year_cycle = np.sin(np.linspace(0, 2 * np.pi, hours))
    
    # Generate features with noise and trends
    temperature = 15 + 10 * year_cycle + 5 * day_cycle + np.random.normal(0, 2, hours)
    humidity = 60 + 15 * year_cycle - 10 * day_cycle + np.random.normal(0, 5, hours)
    humidity = np.clip(humidity, 0, 100)
    
    # Pollutants (higher in winter/night, lower in summer/day)
    # Usually higher when temperature is lower (heating/inversion)
    base_pollution = 30 - 10 * year_cycle + 5 * day_cycle 
    
    pm25 = base_pollution + np.random.normal(15, 8, hours) + 5 * (temperature < 10)
    pm25 = np.clip(pm25, 5, 250) # Non-negative, some spikes
    
    pm10 = pm25 * 1.5 + np.random.normal(5, 4, hours)
    pm10 = np.clip(pm10, 10, 400)
    
    co2 = 400 + 50 * day_cycle + np.random.normal(0, 10, hours)
    
    # Approximate AQI (Simplified)
    # AQI is usually max of individual pollutant indices.
    # Here we just correlate it with PM2.5 for simplicity in this synthetic generation
    aqi = pm25 * 1.8 + np.random.normal(0, 5, hours)
    aqi = np.clip(aqi, 10, 300).astype(int)
    
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Temperature': np.round(temperature, 1),
        'Humidity': np.round(humidity, 1),
        'CO2': np.round(co2, 1),
        'PM10': np.round(pm10, 1),
        'PM2.5': np.round(pm25, 1),
        'AQI': aqi
    })
    
    # Introduce some missing values (5%) to simulate sensor offline
    mask = np.random.choice([True, False], size=hours, p=[0.05, 0.95])
    df.loc[mask, ['PM2.5', 'PM10']] = np.nan
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
    return df

def load_and_clean_data(filepath="data/air_quality.csv"):
    """
    Load data and handle missing values using interpolation/forward fill.
    Returns cleaned dataframe with Timestamp as index.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}. Generating synthetic data...")
        generate_synthetic_data(filepath)
        
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')
    
    # Count missing values before cleaning
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before cleaning: {missing_before}")
    
    # Handle missing values: Time-series interpolation is best
    df = df.interpolate(method='time')
    # If any remain at start/end, forward/backward fill
    df = df.ffill().bfill()
    
    missing_after = df.isnull().sum().sum()
    print(f"Missing values after cleaning: {missing_after}")
    
    return df

def create_sliding_windows(df, window_size=24, forecast_horizon=24, target_col='AQI'):
    """
    Create sliding windows for time-series forecasting.
    window_size: Look-back hours
    forecast_horizon: Predict hours ahead (Currently keeping it 1 for immediate simplicity, but can be updated)
    """
    df_values = df.values
    data_X, data_y = [], []
    
    # We want to predict 'target_col' at t+forecast_horizon
    target_idx = df.columns.get_loc(target_col)
    
    for i in range(len(df_values) - window_size):
        # Input features: all columns from i to i+window_size
        X_window = df_values[i : i + window_size]
        # Target: AQI at i+window_size (immediate next hour for step-by-step)
        # Or predict 24 hours ahead?
        # Let's predict the NEXT hour first for model simplicity, 
        # or forecasting week ahead usually involves recursive prediction or direct multi-output.
        # Let's predict the NEXT hour based on past `window_size` hours.
        y_val = df_values[i + window_size, target_idx]
        
        data_X.append(X_window)
        data_y.append(y_val)
        
    return np.array(data_X), np.array(data_y)

if __name__ == "__main__":
    # Test execution
    df = load_and_clean_data()
    print("Data Preview:\n", df.head())
    
    # Verify sliding windows
    X, y = create_sliding_windows(df, window_size=24, target_col='AQI')
    print(f"X shape: {X.shape} (Windows, Look-back, Features)")
    print(f"y shape: {y.shape}")
