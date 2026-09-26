"""
FastAPI backend for the Knee OA Progression Risk dashboard.
Wraps the existing src.predict.predict_patient_risk() so the static
dashboard (app/static/index.html) can call the real trained model
instead of the client-side demo formula.

Run:
    pip install fastapi uvicorn
    uvicorn app.api:app --reload --port 8000
Then open http://localhost:8000
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predict import predict_patient_risk, load_model_pipeline, list_available_models

app = FastAPI(title="Knee OA Progression Risk API")

# Allow the dashboard to call this API even if served from a different origin
# during local development (e.g. opening the HTML file directly).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None


@app.on_event("startup")
def _load_pipeline():
    global _pipeline
    try:
        _pipeline = load_model_pipeline()
    except FileNotFoundError:
        _pipeline = None  # /api/predict will report this clearly


class PatientInput(BaseModel):
    V00AGE: float = Field(..., ge=18, le=110)
    V00SEX: int = Field(..., ge=1, le=2)
    V00BMI: float = Field(..., ge=10, le=70)
    V00WOMKP: float = Field(..., ge=0, le=20)
    V00WOMAD: float = Field(..., ge=0, le=68)
    V00WOMST: float = Field(..., ge=0, le=8)
    V00PASE: float = Field(..., ge=0, le=500)
    V00KL: int = Field(..., ge=0, le=3)
    V00INJ: int = Field(..., ge=0, le=1)
    V00SURG: int = Field(..., ge=0, le=1)


@app.post("/api/predict")
def predict(patient: PatientInput, model: str | None = None):
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python -m src.train` first, then restart the API.",
        )
    return predict_patient_risk(patient.dict(), _pipeline, model_name=model)


@app.get("/api/models")
def get_models():
    if _pipeline is None:
        return {"models": []}
    return {"models": list_available_models(_pipeline)}


@app.get("/api/health")
def health():
    return {"model_loaded": _pipeline is not None}


# Serve the dashboard itself at "/"
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")