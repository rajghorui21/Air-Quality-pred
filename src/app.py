from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta

app = FastAPI(title="Smart Air Quality Prediction")

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Mount Static Files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load Model and Data on Startup
MODEL_PATH = os.path.join(BASE_DIR, "models", "aqi_xgb_model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "air_quality.csv")

model = None
metadata = None
df_cleaned = None

@app.on_event("startup")
def load_assets():
    global model, metadata, df_cleaned
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded.")
    if os.path.exists(METADATA_PATH):
        metadata = joblib.load(METADATA_PATH)
        print("Metadata loaded.")
    if os.path.exists(DATA_PATH):
        # Load and clean data using data_utils logic inline to avoid relative import issues
        # Or just read CSV and interpolate
        df = pd.read_csv(DATA_PATH)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.set_index('Timestamp')
        df = df.interpolate(method='time').ffill().bfill()
        df_cleaned = df
        print("Data loaded and cleaned.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/dashboard_data")
def get_dashboard_data():
    if df_cleaned is None or model is None:
        return {"error": "Model or Data not loaded."}
    
    # Get last 48 hours for historical view
    hist_hours = 48
    hist_data = df_cleaned.tail(hist_hours).reset_index()
    hist_data['Timestamp'] = hist_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    current_state = df_cleaned.iloc[-1]
    last_timestamp = df_cleaned.index[-1]
    
    # Prepare recursive forecast for next 24 hours
    window_size = metadata.get('window_size', 24)
    # Get last `window_size` hours as input
    input_window = df_cleaned.tail(window_size).values # Shape (24, 6)
    
    forecast_timestamps = []
    forecast_values = []
    
    current_input = input_window.copy()
    
    for i in range(24):
        # Flatten input for model: (1, 144)
        input_flat = current_input.reshape(1, -1)
        pred_aqi = model.predict(input_flat)[0]
        
        # Next timestamp
        next_time = last_timestamp + timedelta(hours=i+1)
        forecast_timestamps.append(next_time.strftime('%Y-%m-%d %H:%M'))
        forecast_values.append(float(pred_aqi))
        
        # Update current_input for next iteration (Recursive)
        # Shift window up
        next_row = current_input[-1].copy() # Copy last row to get scale
        # Update AQI in next row (Target prediction)
        # Important: We need to know which index is AQI
        aqi_idx = list(df_cleaned.columns).index('AQI')
        next_row[aqi_idx] = pred_aqi
        # For other features (Temp, Hum, etc.), for recursive purely on inputs, 
        # normally we'd need forecasts for them too, or assume static/cycle.
        # To keep it simple and robust for demo, we can just roll the window 
        # and keep other features relatively same or add small noise, 
        # OR just use the last known values.
        # Let's shift the window and append the new row with predicted AQI.
        # For other features, let's keep them same as the last step to simulate "holding steady" 
        # or propagate them.
        
        current_input = np.roll(current_input, -1, axis=0)
        current_input[-1] = next_row # Update the new row at the bottom
        
    return {
        "current": {
            "aqi": int(current_state['AQI']),
            "temperature": current_state['Temperature'],
            "humidity": current_state['Humidity'],
            "timestamp": last_timestamp.strftime('%Y-%m-%d %H:%M')
        },
        "historical": {
            "timestamps": hist_data['Timestamp'].tolist(),
            "aqi": hist_data['AQI'].tolist(),
            "pm25": hist_data['PM2.5'].tolist()
        },
        "forecast": {
            "timestamps": forecast_timestamps,
            "aqi": [int(v) for v in forecast_values]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
