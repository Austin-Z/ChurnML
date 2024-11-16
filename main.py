from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import requests
import os
from typing import List, Dict
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

        # Step 1: Keep a copy of the original CustomerID
        df['Original_CustomerID'] = df['CustomerID']
        df['CLTV'] = df['CLTV']
        # Step 2: Convert CustomerID to a unique integer identifier for model input
        df['CustomerID'] = pd.factorize(df['CustomerID'])[0]
        
        df['ChargesPerMonth'] = df['Total Charges'] / df['Tenure Months']
        
        # Drop unnecessary columns
        df = df.drop(['Total Charges', 'Monthly Charges'], axis=1)
        # Identify columns that need one-hot encoding (exclude Original_CustomerID)
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = categorical_columns.drop(['Original_CustomerID'])  # Exclude specific columns
        
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        # One-Hot Encoding for all categorical columns, dropping the first category

        model_features = model.feature_names_in_
        missing_features = [col for col in model_features if col not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Input data is missing required feature columns: {', '.join(missing_features)}"
            )
        # Separate features for model prediction
        df_features = df.reindex(columns=model_features, fill_value=0)
        
        # Predict churn probabilities
        churn_probabilities = model.predict_proba(df_features)[:, 1]
        
        # Prepare response
        results = sorted(
            [
                {
                    "Original_CustomerID": str(original_id),  # Original CustomerID for display
                    "CLTV": float(cltv),
                    "ChargesPerMonth": float(charges_per_month),
                    "Churn Probabilities": float(prob)
                }
                for original_id, cltv, charges_per_month, prob in zip(
                    df['Original_CustomerID'], df['CLTV'], df['ChargesPerMonth'], churn_probabilities
                )
            ],
            key=lambda x: x['Churn Probabilities'],
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
