
import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any


# --------------------------------------------------
# 1. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Machine Learning API for Customer Churn Prediction",
    version="1.0"
)


# --------------------------------------------------
# 2. Load trained model and label encoder
# --------------------------------------------------

model = joblib.load("best_churn_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# --------------------------------------------------
# 3. Define original dataset columns
# --------------------------------------------------

RAW_COLUMNS = [
    "Gender",
    "Senior_Citizen",
    "Tenure_Months",
    "Contract_Type",
    "Internet_Service",
    "Payment_Method",
    "Monthly_Charges_INR",
    "Total_Charges_INR",
    "Tech_Support",
    "Online_Security",
    "Support_Tickets_6M",
    "Avg_Data_Usage_GB"
]


# IMPORTANT:
# Replace 10.0 with the actual median value of
# Avg_Data_Usage_GB calculated in your training notebook.

USAGE_THRESHOLD = 10.0


# --------------------------------------------------
# 4. Define model feature columns
# --------------------------------------------------

MODEL_FEATURES = [
    "Gender",
    "Senior_Citizen",
    "Tenure_Months",
    "Contract_Type",
    "Internet_Service",
    "Payment_Method",
    "Monthly_Charges_INR",
    "Total_Charges_INR",
    "Tech_Support",
    "Online_Security",
    "Support_Tickets_6M",
    "Avg_Data_Usage_GB",
    "Tenure_Category",
    "Average_Monthly_Spending",
    "Has_Internet_Service",
    "Has_Tech_Support",
    "Has_Online_Security",
    "Ticket_Intensity",
    "High_Data_Usage"
]


# --------------------------------------------------
# 5. Request data format
# --------------------------------------------------

class CustomerData(BaseModel):
    customer: Dict[str, Any]


# --------------------------------------------------
# 6. Feature engineering
# --------------------------------------------------

def create_features(data):

    df = pd.DataFrame([data])

    numeric_columns = [
        "Tenure_Months",
        "Monthly_Charges_INR",
        "Total_Charges_INR",
        "Support_Tickets_6M",
        "Avg_Data_Usage_GB"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Handle invalid values

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
        df.loc[df[col] < 0, col] = np.nan

    # Tenure category

    df["Tenure_Category"] = pd.cut(
        df["Tenure_Months"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "New",
            "Short-term",
            "Medium-term",
            "Long-term"
        ]
    )

    # Average monthly spending

    df["Average_Monthly_Spending"] = np.where(
        df["Tenure_Months"] > 0,
        df["Total_Charges_INR"] / df["Tenure_Months"],
        df["Monthly_Charges_INR"]
    )

    # Internet service indicator

    internet = (
        df["Internet_Service"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["Has_Internet_Service"] = internet.isin(
        ["yes", "fiber optic", "dsl", "true", "1"]
    ).astype(int)

    # Technical support indicator

    tech = (
        df["Tech_Support"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["Has_Tech_Support"] = tech.isin(
        ["yes", "true", "1"]
    ).astype(int)

    # Online security indicator

    security = (
        df["Online_Security"]
        .astype(str)
        .str.strip()
        .str.lower()
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

    # High data usage indicator

    df["High_Data_Usage"] = (
        df["Avg_Data_Usage_GB"] > USAGE_THRESHOLD
    ).astype(int)

    # Match model training columns and order

    return df.reindex(columns=MODEL_FEATURES)


# --------------------------------------------------
# 7. Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# --------------------------------------------------
# 8. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(customer_data: CustomerData):

    try:

        customer = customer_data.customer

        # Check missing input fields

        missing_columns = [
            col for col in RAW_COLUMNS
            if col not in customer
        ]

        if missing_columns:

            raise HTTPException(
                status_code=400,
                detail=f"Missing fields: {missing_columns}"
            )

        # Generate features

        features = create_features(customer)

        # Predict churn

        prediction = model.predict(features)[0]

        churn_status = label_encoder.inverse_transform(
            [int(prediction)]
        )[0]

        return {
            "prediction": str(churn_status),
            "message": (
                "Customer is likely to churn"
                if str(churn_status).lower() == "churn"
                else "Customer is likely to stay"
            )
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
