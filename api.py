# Load saved model
model = joblib.load("best_churn_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define metadata directly
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

USAGE_THRESHOLD = 10.0  # Replace with the actual training median

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
