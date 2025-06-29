import os, sys
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel, ValidationError
from datetime import datetime
from typing import Optional
import uvicorn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, KBinsDiscretizer
import numpy as np
from datetime import datetime
# Add the 'scripts' directory to the Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join('..', 'scripts')))
from scripts.feature_engg import FeatureEngineering # type: ignore
from scripts.credit_scoring_model import CreditScoreRFM # type: ignore
import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
  import gdown # type: ignore
except ImportError:
  print("Error: gdown package not found. Please ensure gdown is in your requirements.txt file")
  gdown = None

def download_file_from_google_drive(file_id, destination):
    """
    Downloads a file from Google Drive using gdown.

    Args:
        file_id (str): The Google Drive file ID.
        destination (str): The local file path where the downloaded file will be saved.
    """
    if gdown is not None:
        url = f"https://drive.google.com/uc?id={file_id}"
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        gdown.download(url, destination, quiet=False)
    else:
      raise Exception("gdown was not imported correctly. Please make sure it is in the requirements.txt file")
    


def load_model(model_path='model/best_model.pkl'):
    """Loads the model from local storage."""
    
    #
    if not os.path.exists(model_path):
        file_id = '1WFVrXo2AtP13yRAoiYHekTyuv2BDMsY2' # Replace with your actual file ID from Google Drive
        try:
             download_file_from_google_drive(file_id, model_path)
        except Exception as e:
             raise FileNotFoundError(f"Could not download the model from Google Drive. Error: {e}")   
    model = joblib.load(model_path)
    print(type(model))
    return model



app = FastAPI()
model = load_model()

# Input schema
class InputData(BaseModel):
    TransactionId: int
    CustomerId: int
    ProductCategory: int
    ChannelId: str
    Amount: float
    TransactionStartTime: datetime
    PricingStrategy: int


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Credit Scoring API"}


@app.post("/predict")
async def predict(input_data: InputData):
    try:
        # Log received data
        logging.info(f"Received input data: {input_data}")

        # Prepare input data as a DataFrame
        input_data_dict = {
            "TransactionId": input_data.TransactionId,
            "CustomerId": input_data.CustomerId,
            "ProductCategory": input_data.ProductCategory,
            "ChannelId": input_data.ChannelId,
            "Amount": input_data.Amount,
            "TransactionStartTime": input_data.TransactionStartTime,
            "PricingStrategy": input_data.PricingStrategy,
        }
        input_df = pd.DataFrame([input_data_dict])

        # Log preprocessing start
        logging.info("Starting feature engineering...")

        # Feature Engineering
        fe = FeatureEngineering()
        input_df = fe.create_aggregate_features(input_df)
        input_df = fe.create_transaction_features(input_df)
        input_df = fe.extract_time_features(input_df)

        # Encode categorical features
        categorical_cols = ["ProductCategory", "ChannelId"]
        input_df = fe.encode_categorical_features(input_df, categorical_cols)

        # Normalize numerical features
        numeric_cols = input_df.select_dtypes(include="number").columns.tolist()
        exclude_cols = ["Amount", "TransactionId"]
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        input_df = fe.normalize_numerical_features(
            input_df, numeric_cols, method="standardize"
        )

        # RFM Calculation
        rfm = CreditScoreRFM(input_df.reset_index())
        rfm_df = rfm.calculate_rfm()

        # Merge RFM features with the input data
        final_df = pd.merge(input_df, rfm_df, on="CustomerId", how="left")

        # Ensure all required features exist in the final_df
        required_features = [
            "ProductCategory",
            "PricingStrategy",
            "Transaction_Count",
            "Transaction_Month",
            "Transaction_Year",
            "Recency",
            "Frequency",
        ]

        # Fill missing features with default values (e.g., 0 for numerical features)
        for feature in required_features:
            if feature not in final_df.columns:
                final_df[feature] = 0  # Fill with default value

        # Reindex the final_df to match the training feature order
        final_df = final_df.reindex(columns=required_features, fill_value=0)

        # Log prediction start
        logging.info("Making prediction...")

        # Make prediction
        prediction = model.predict(final_df)
        predicted_risk = "Good" if prediction[0] == 0 else "Bad"

        # Log prediction result
        logging.info(
            f"Prediction complete: Customer ID {input_data.CustomerId}, Predicted Risk: {predicted_risk}"
        )

        # Return response
        return {"customer_id": input_data.CustomerId, "predicted_risk": predicted_risk}

    except ValidationError as ve:
        logging.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=f"Validation error: {ve}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred: {str(e)}"
        )