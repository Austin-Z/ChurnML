from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model import predict  # Import the predict function from model.py
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Add CORS middleware to allow requests from any origin (for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; adjust if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Define the input schema
class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.post("/predict")
def get_prediction(input_data: ModelInput):
    # Convert input data to list format that `predict` function expects
    features = [input_data.feature1, input_data.feature2, input_data.feature3]
    prediction_result = predict(features)  # Call the predict function with input features

    return {"prediction": prediction_result}
