
import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0"
)

# Load saved model and metadata
model = joblib.load("best_churn_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


RAW_COLUMNS = metadata["raw_columns"]
USAGE_THRESHOLD = metadata["usage_threshold"]
MODEL_FEATURES = metadata["model_features"]


class CustomerData(BaseModel):
    customer: Dict[str, Any]


def create_features(data):
    df = pd.DataFrame([data])

    # Convert numeric columns
    numeric_columns = [
        "Tenure_Months",
        "Monthly_Charges_INR",
        "Total_Charges_INR",
        "Support_Tickets_6M",
        "Avg_Data_Usage_GB"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle invalid values consistently with training
    if "Tenure_Months" in df.columns:
        df.loc[
            (df["Tenure_Months"] < 0) |
            (df["Tenure_Months"] > 72),
            "Tenure_Months"
        ] = np.nan

    for col in [
        "Monthly_Charges_INR",
        "Total_Charges_INR",
        "Support_Tickets_6M",
        "Avg_Data_Usage_GB"
    ]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # Tenure category
    df["Tenure_Category"] = pd.cut(
        df["Tenure_Months"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New", "Short-term", "Medium-term", "Long-term"]
    )

    # Average monthly spending
    df["Average_Monthly_Spending"] = np.where(
        df["Tenure_Months"] > 0,
        df["Total_Charges_INR"] / df["Tenure_Months"],
        df["Monthly_Charges_INR"]
    )

    # Service indicators
    internet = (
        df["Internet_Service"].astype(str).str.strip().str.lower()
    )
    df["Has_Internet_Service"] = internet.isin(
        ["yes", "fiber optic", "dsl", "true", "1"]
    ).astype(int)

    tech = df["Tech_Support"].astype(str).str.strip().str.lower()
    df["Has_Tech_Support"] = tech.isin(
        ["yes", "true", "1"]
    ).astype(int)

    security = (
        df["Online_Security"].astype(str).str.strip().str.lower()
    )
    df["Has_Online_Security"] = security.isin(
        ["yes", "true", "1"]
    ).astype(int)

    # Ticket intensity
    df["Ticket_Intensity"] = np.where(
        df["Tenure_Months"] > 0,
        df["Support_Tickets_6M"] / df["Tenure_Months"],
        df["Support_Tickets_6M"]
    )

    # High data usage
    df["High_Data_Usage"] = (
        df["Avg_Data_Usage_GB"] > USAGE_THRESHOLD
    ).astype(int)

    # Match training feature order
    return df.reindex(columns=MODEL_FEATURES)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(customer_data: CustomerData):

    try:
        customer = customer_data.customer

        missing_columns = [
            col for col in RAW_COLUMNS
            if col not in customer
        ]

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing fields: {missing_columns}"
            )

        features = create_features(customer)

        prediction = model.predict(features)[0]
        churn_status = label_encoder.inverse_transform(
            [int(prediction)]
        )[0]

        return {
            "prediction": str(churn_status),
            "message": "Customer is likely to churn"
            if str(churn_status).lower() == "churn"
            else "Customer is likely to stay"
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
