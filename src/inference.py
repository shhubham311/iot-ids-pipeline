import sys
import os
import json
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
import uvicorn

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- GLOBAL VARIABLES ---
# We load these once when the server starts to save time on every request
model = None
feature_encoders = None
target_encoder = None

app = FastAPI(title="IoT Intrusion Detection API", version="1.0")

class TrafficInput(BaseModel):
    """
    Input format: A dictionary of all feature values.
    We use Dict[str, Any] to accept the 80+ numerical columns 
    plus 'proto' and 'service'.
    """
    data: Dict[str, Any]

def load_artifacts():
    """
    Loads the latest model and encoders from disk.
    """
    global model, feature_encoders, target_encoder
    
    print("Loading artifacts...")
    
    # Calculate path to models folder
    current_dir = os.path.dirname(os.path.abspath(__file__)) # /app/src
    project_root = os.path.dirname(current_dir)              # /app
    models_dir = os.path.join(project_root, "models")        # /app/models
    
    # 1. Load Encoders
    feature_encoders = joblib.load(os.path.join(models_dir, "feature_encoders.pkl"))
    target_encoder = joblib.load(os.path.join(models_dir, "target_encoder.pkl"))
    print("Encoders loaded.")

    # 2. Load Model (Directly from file - No MLflow server needed)
    model = joblib.load(os.path.join(models_dir, "xgboost_model.pkl"))
    print("Model loaded successfully.")

# Load artifacts on startup
@app.on_event("startup")
def startup_event():
    load_artifacts()

@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(input_data: TrafficInput):
    """
    Accepts raw traffic features, predicts Attack Type.
    """
    try:
        # 1. Convert input dict to DataFrame
        # The input is expected to be one row of data
        df = pd.DataFrame([input_data.data])
        
        # 2. Encode Categorical Features (Same as training)
        # We iterate through the keys in our saved encoders
        # e.g., 'le_proto', 'le_service'
        for key, encoder in feature_encoders.items():
            # Extract column name from key (e.g., 'le_proto' -> 'proto')
            col_name = key.replace("le_", "")
            
            if col_name in df.columns:
                # Handle unseen categories gracefully (assign -1)
                df[col_name] = df[col_name].apply(lambda x: x if x in encoder.classes_ else -1)
                
                # Transform known ones to integers
                mask = df[col_name] != -1
                df.loc[mask, col_name] = encoder.transform(df.loc[mask, col_name])
                
                # Force integer type
                df[col_name] = df[col_name].astype(int)
            else:
                # If column missing in input, fill with 0
                df[col_name] = 0

        # 3. Ensure all columns match model training
        # XGBoost throws error if column names differ
        # We use model.get_booster().feature_names to align
        model_features = model.get_booster().feature_names
        
        # Reindex df to match model features (fill missing with 0)
        df = df.reindex(columns=model_features, fill_value=0)

        # 4. Predict
        prediction_id = model.predict(df)[0]
        
        # 5. Decode Label
        attack_name = target_encoder.inverse_transform([prediction_id])[0]
        
        return {
            "prediction_id": int(prediction_id),
            "attack_type": attack_name,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)