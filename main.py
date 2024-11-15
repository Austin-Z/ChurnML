from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

# Enable CORS for all origins 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
try:
    model = joblib.load(r'c:\Users\austi\Github\ChurnML\calibrated_rf_model.joblib')
except Exception as e:
    print("Error loading model:", e)
    model = None

@app.post("/predict-churn")
async def predict_churn(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})

    try:
        # Read the uploaded CSV file into a pandas DataFrame
        df = pd.read_csv(file.file)

        # Rename columns if necessary
        if 'CustomerID' in df.columns:
            df = df.rename(columns={'CustomerID': 'user_id'})
        if 'CLTV_value' in df.columns:
            df = df.rename(columns={'CLTV_value': 'CLTV'})

        required_columns = ['user_id', 'CLTV', 'Reason']
        if not all(col in df.columns for col in required_columns):
            return JSONResponse(status_code=400, content={"error": f"CSV file must contain columns: {', '.join(required_columns)}"})

        # Extract necessary columns
        user_ids = df['user_id']
        cltv_values = df['CLTV']
        reasons = df['Reason']

        # Preprocess for model input and make predictions
        model_features = model.feature_names_in_
        df_features = df.reindex(columns=model_features, fill_value=0)
        churn_probabilities = model.predict_proba(df_features)[:, 1]

        # Prepare sorted results
        results = sorted(
            [
                {"user_id": uid, "cltv": cltv, "reason": reason, "churn_probability": prob}
                for uid, cltv, reason, prob in zip(user_ids, cltv_values, reasons, churn_probabilities)
            ],
            key=lambda x: x['churn_probability'],
            reverse=True  # Sort in descending order
        )

        return {"predictions": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


