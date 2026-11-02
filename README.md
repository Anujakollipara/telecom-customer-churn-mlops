
# Customer Churn Prediction System

## Project Description
A Machine Learning-based system that predicts whether a
telecommunications customer is likely to churn or not churn.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- FastAPI
- GitHub Actions
- Docker
- GitHub Container Registry

## Machine Learning Models
1. Logistic Regression
2. Random Forest

## API Endpoints
- GET /health
- POST /predict

## Model Files
- best_churn_model.pkl
- label_encoder.pkl
- api_metadata.pkl

## Installation
pip install -r requirements.txt

## Run Application
uvicorn api:app --host 0.0.0.0 --port 8000

## API Documentation
http://localhost:8000/docs
