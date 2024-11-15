from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import requests
import os
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ChurnML Predictor")

# Enable CORS for all origins 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for model
model = None

@app.on_event("startup")
async def load_model():
    """Load model on startup instead of module level"""
    global model
    try:
        model_url = 'https://ML.jampajoy.com/calibrated_rf_model.joblib'
        response = requests.get(model_url, timeout=30)  # Add timeout
        response.raise_for_status()  # Raise error for bad status codes
        
        with open('calibrated_rf_model.joblib', 'wb') as f:
            f.write(response.content)
        
        model = joblib.load('calibrated_rf_model.joblib')
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/predict-churn")
async def predict_churn(file: UploadFile = File(...)) -> Dict:
    """
    Predict churn probability from uploaded CSV file
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read the uploaded CSV file into a pandas DataFrame
        df = pd.read_csv(file.file)
        
        # Column renaming
        column_mapping = {
            'CustomerID': 'user_id',
            'CLTV_value': 'CLTV'
        }
        df = df.rename(columns=column_mapping)
        
        # Validate required columns
        required_columns = ['user_id', 'CLTV', 'Reason']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Extract necessary columns
        user_ids = df['user_id']
        cltv_values = df['CLTV']
        reasons = df['Reason']
        
        # Preprocess for model input and make predictions
        model_features = model.feature_names_in_
        df_features = df.reindex(columns=model_features, fill_value=0)
        
        # Validate feature columns
        if not all(col in df.columns for col in model_features):
            raise HTTPException(
                status_code=400,
                detail="Input data missing required feature columns"
            )
        
        churn_probabilities = model.predict_proba(df_features)[:, 1]
        
        # Prepare sorted results
        results = sorted(
            [
                {
                    "user_id": str(uid),  # Convert to string to ensure JSON serializable
                    "cltv": float(cltv),
                    "reason": str(reason),
                    "churn_probability": float(prob)
                }
                for uid, cltv, reason, prob in zip(user_ids, cltv_values, reasons, churn_probabilities)
            ],
            key=lambda x: x['churn_probability'],
            reverse=True
        )
        
        return {"predictions": results}
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
