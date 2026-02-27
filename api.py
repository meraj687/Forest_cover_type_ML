from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI(title="Forest Cover Prediction API")

# ---------------- LOAD MODEL ----------------
with open("rfc.pkl", "rb") as f:
    model = pickle.load(f)

EXPECTED_FEATURES = model.n_features_in_

# ---------------- REQUEST SCHEMA ----------------
class PredictionRequest(BaseModel):
    features: list


# ---------------- MODEL INFO ENDPOINT ----------------
@app.get("/model-info")
def model_info():
    return {
        "expected_features": EXPECTED_FEATURES,
        "feature_names": model.feature_names_in_.tolist()
        if hasattr(model, "feature_names_in_")
        else [f"Feature_{i+1}" for i in range(EXPECTED_FEATURES)]
    }


# ---------------- PREDICT ENDPOINT ----------------
@app.post("/predict")
def predict(data: PredictionRequest):

    values = data.features

    if len(values) != EXPECTED_FEATURES:
        return {
            "error": f"Model expects {EXPECTED_FEATURES} features."
        }

    features = np.array(values).reshape(1, -1)

    prediction = int(model.predict(features)[0])

    if hasattr(model, "predict_proba"):
        probability = float(np.max(model.predict_proba(features)) * 100)
    else:
        probability = None

    return {
        "prediction": prediction,
        "confidence": probability
    }


# ---------------- FEATURE IMPORTANCE ----------------
@app.get("/feature-importance")
def feature_importance():

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_.tolist()

        if hasattr(model, "feature_names_in_"):
            names = model.feature_names_in_.tolist()
        else:
            names = [f"Feature_{i+1}" for i in range(len(importance))]

        return {
            "feature_names": names,
            "importance": importance
        }

    return {"error": "Model does not support feature importance"}