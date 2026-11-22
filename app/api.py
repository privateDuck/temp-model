import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from inference import CITY_COORDS, engineer_features, fetch_recent_open_meteo

DEFAULT_MODEL_URI = "models:/openmeteo-forecast/latest"
MODEL: dict = {"pyfunc": None, "uri": None}


"""Load the pyfunc model once when uvicorn starts (blocks serving
until ready, so the dashboard never hits a half-initialised API)."""
@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")
    model_uri = os.environ.get("MLFLOW_MODEL_URI", DEFAULT_MODEL_URI)
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[api] loading {model_uri} from {tracking_uri} ...")
    MODEL["pyfunc"] = mlflow.pyfunc.load_model(model_uri)
    MODEL["uri"] = model_uri
    print("[api] model ready")
    yield
    MODEL["pyfunc"] = None


app = FastAPI(title="6h Temperature Forecast", lifespan=lifespan)


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    city: Optional[str] = None  # override the city label used for one-hot


# Nearest training city if within ~1 degree, else 'custom' (unknown cities predict with all one-hot flags zero)
def _resolve_city(lat: float, lon: float, city: Optional[str]) -> str:
    if city:
        return city
    best = min(CITY_COORDS,
               key=lambda c: (CITY_COORDS[c][0] - lat) ** 2
               + (CITY_COORDS[c][1] - lon) ** 2)
    dlat = abs(CITY_COORDS[best][0] - lat)
    dlon = abs(CITY_COORDS[best][1] - lon)
    return best if max(dlat, dlon) <= 1.0 else "custom"


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL["pyfunc"] is not None else "loading",
        "model_uri": MODEL["uri"],
    }


@app.post("/predict")
def predict(req: PredictRequest):
    model = MODEL["pyfunc"]
    if model is None:
        raise HTTPException(503, "model is still loading, try again shortly")

    t0 = time.perf_counter()

    try:
        raw = fetch_recent_open_meteo(req.latitude, req.longitude)
    except Exception as exc:
        raise HTTPException(502, f"Open-Meteo fetch failed: {exc}")
    fetch_ms = (time.perf_counter() - t0) * 1000

    # 2. features, keep only hours already observed, latest hour last
    city = _resolve_city(req.latitude, req.longitude, req.city)
    try:
        feats = engineer_features(raw, city, req.latitude, req.longitude)
    except Exception as exc:
        raise HTTPException(502, f"feature engineering failed: {exc}")
    feats = feats[feats["timestamp"] <= pd.Timestamp.now(tz="UTC")]
    row = feats.tail(1)

    t1 = time.perf_counter()
    pred = float(np.asarray(model.predict(row), dtype=float)[0])
    if not np.isfinite(pred):
        for k in range(2, 7):
            row = feats.tail(k).head(1)
            pred = float(np.asarray(model.predict(row), dtype=float)[0])
            if np.isfinite(pred):
                break
    inference_ms = (time.perf_counter() - t1) * 1000

    if not np.isfinite(pred):
        raise HTTPException(
            502, "prediction is NaN for the last 6 observed hours")

    observed = row["timestamp"].iloc[0]
    return {
        "city": city,
        "latitude": req.latitude,
        "longitude": req.longitude,
        "observed_at": observed.isoformat(),
        "temperature_now_c": round(float(row["temperature_2m"].iloc[0]), 2),
        "forecast_time": (observed + pd.Timedelta(hours=6)).isoformat(),
        "predicted_temperature_c": round(pred, 2),
        "inference_ms": round(inference_ms, 1),
        "fetch_ms": round(fetch_ms, 1),
        "total_ms": round((time.perf_counter() - t0) * 1000, 1),
    } 