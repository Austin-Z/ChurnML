# main.py
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
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

# Database Configuration
DATABASE_URL = f"mysql+pymysql://{os.getenv('AZURE_MYSQL_USER')}:{os.getenv('AZURE_MYSQL_PASSWORD')}@{os.getenv('AZURE_MYSQL_HOST')}:3306/{os.getenv('AZURE_MYSQL_NAME')}"

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define a model for storing predictions (optional)
class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    feature1 = Column(Float, nullable=False)
    feature2 = Column(Float, nullable=False)
    feature3 = Column(Float, nullable=False)
    result = Column(String, nullable=False)

# Create the table (only needed once)
Base.metadata.create_all(bind=engine)

# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Define the input schema
class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.post("/predict")
def get_prediction(input_data: ModelInput, db: Session = Depends(get_db)):
    # Convert input data to list format that `predict` function expects
    features = [input_data.feature1, input_data.feature2, input_data.feature3]
    prediction_result = predict(features)  # Call the predict function with input features

    # Optionally, save the prediction to the database
    db_prediction = Prediction(
        feature1=input_data.feature1,
        feature2=input_data.feature2,
        feature3=input_data.feature3,
        result=str(prediction_result)
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)

    return {"prediction": prediction_result, "prediction_id": db_prediction.id}
