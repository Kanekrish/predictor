import os
import tempfile
import joblib
import pandas as pd
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Breast Cancer Stage Predictor API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_URL = os.getenv("MODEL_URL", "https://raw.githubusercontent.com/Kanekrish/predictor/main/model/xgboost_model.pkl")
TARGET_ENCODER_URL = os.getenv(
    "TARGET_ENCODER_URL",
    "https://raw.githubusercontent.com/Kanekrish/predictor/main/model/target_encoder.pkl"
)
FEATURE_ENCODERS_URL = os.getenv(
    "FEATURE_ENCODERS_URL",
    "https://raw.githubusercontent.com/Kanekrish/predictor/main/model/feature_encoders.pkl"
)

_feature_columns = None

def load_pickle_from_github(url, filename):
    if not url or url.startswith("https://raw.githubusercontent.com/Kanekrish/predictor/main/model/xgboost_model.pkl"):
        raise RuntimeError(
            f"Set the GitHub URL for {filename} before deploying."
        )

    cache_dir = os.path.join(tempfile.gettempdir(), "bc_stage_model")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)

    if not os.path.exists(path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)

    return joblib.load(path)

def get_artifacts():
    model = load_pickle_from_github(
        MODEL_URL, "https://raw.githubusercontent.com/Kanekrish/predictor/main/model/xgboost_model.pkl"
    )
    target_encoder = load_pickle_from_github(
        TARGET_ENCODER_URL, "https://raw.githubusercontent.com/Kanekrish/predictor/main/model/target_encoder.pkl"
    )
    feature_encoders = load_pickle_from_github(
        FEATURE_ENCODERS_URL, "https://raw.githubusercontent.com/Kanekrish/predictor/main/model/feature_encoders.pkl"
    )
    return model, target_encoder, feature_encoders

class PredictionRequest(BaseModel):
    data: dict

@app.get("/api")
def health():
    return {
        "status": "online",
        "message": "Breast Cancer Stage Predictor API"
    }

@app.post("/api/predict")
def predict(request: PredictionRequest):
    try:
        model, target_encoder, feature_encoders = get_artifacts()

        row = {
            str(k).strip(): str(v)
            for k, v in request.data.items()
        }

        expected_columns = list(feature_encoders.keys())

        missing = [
            c for c in expected_columns
            if c not in row
        ]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing feature(s): {missing}"
            )

        input_df = pd.DataFrame(
            [[row[c] for c in expected_columns]],
            columns=expected_columns
        )

        for col in expected_columns:
            encoder = feature_encoders[col]
            value = input_df.at[0, col]

            if value not in encoder.classes_:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Value '{value}' is not recognised for "
                        f"feature '{col}'."
                    )
                )

            input_df[col] = encoder.transform(
                input_df[col].astype(str)
            )

        probabilities = model.predict_proba(input_df.values)[0]
        prediction = int(model.predict(input_df.values)[0])

        predicted_stage = target_encoder.inverse_transform(
            [prediction]
        )[0]

        return {
            "predicted_stage": str(predicted_stage),
            "confidence_percent": round(
                float(probabilities.max()) * 100, 2
            )
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )
