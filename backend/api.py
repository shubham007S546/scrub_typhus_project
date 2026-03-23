"""
Scrub Typhus Prediction API
Industry-level FastAPI backend with:
  - Strong Pydantic v2 models with clinical validation
  - Random Forest prediction + XAI (SHAP-equivalent via feature importance)
  - Treatment recommendations engine
  - Health / readiness endpoints
  - Structured logging
  - CORS for frontend
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

# ─── Logging Setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("scrub_typhus_api")

# ─── Path resolution ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "ml"
DATA_DIR = BASE_DIR / "data"
# ─── Load model & metadata ──────────────────────────────────────────────────
logger.info("Loading ML pipeline...")
PIPELINE = joblib.load(ML_DIR / "model_pipeline.pkl")
with open(ML_DIR / "model_metrics.json") as f:
    MODEL_METRICS = json.load(f)

FEATURE_NAMES: List[str] = MODEL_METRICS["feature_names"]
FEATURE_IMPORTANCE: Dict[str, float] = MODEL_METRICS["feature_importance"]
logger.info("ML pipeline loaded. Accuracy=%.4f  ROC-AUC=%.4f",
            MODEL_METRICS["accuracy"], MODEL_METRICS["roc_auc"])

# ─── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Scrub Typhus AI Prediction API",
    description=(
        "Industry-grade REST API for Scrub Typhus (Orientia tsutsugamushi) "
        "prediction using Random Forest + Explainable AI. "
        "Provides diagnosis probability, risk stratification, and treatment recommendations."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request tracking middleware ────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    req_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    logger.info("[%s] %s %s → %d (%.1f ms)", req_id, request.method,
                request.url.path, response.status_code, elapsed)
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    return response

# ─── Pydantic Models ────────────────────────────────────────────────────────

class PatientInput(BaseModel):
    """
    Clinical input for Scrub Typhus prediction.
    All validators are grounded in clinical reference ranges.
    """
    # ── Demographics ─────────────────────────────────────────────────────
    age: int = Field(..., ge=1, le=120, description="Patient age in years", example=35)
    gender: int = Field(..., ge=0, le=1, description="0=Male, 1=Female", example=1)
    rural_background: int = Field(..., ge=0, le=1, description="Rural background (1=Yes)", example=1)
    occupation_risk: int = Field(..., ge=0, le=1, description="High-risk occupation: farmer/housewife (1=Yes)", example=1)

    # ── Environmental ────────────────────────────────────────────────────
    monsoon_season: int = Field(..., ge=0, le=1, description="Presentation in Jul-Sep monsoon season", example=1)
    altitude_hills: int = Field(..., ge=0, le=1, description="Hilly/sub-Himalayan region residence", example=1)
    rainfall_mm: float = Field(..., ge=0.0, le=600.0, description="Recent local rainfall in mm", example=180.0)
    humidity_pct: float = Field(..., ge=0.0, le=100.0, description="Ambient humidity percentage", example=75.0)
    temp_celsius: float = Field(..., ge=-10.0, le=50.0, description="Ambient temperature °C", example=28.0)
    outdoor_exposure: int = Field(..., ge=0, le=1, description="Recent outdoor/field exposure (1=Yes)", example=1)

    # ── Clinical Symptoms ────────────────────────────────────────────────
    fever_days: int = Field(..., ge=1, le=30, description="Duration of fever in days", example=8)
    fever_temp_c: float = Field(..., ge=35.0, le=42.5, description="Maximum recorded fever temperature °C", example=39.2)
    headache: int = Field(..., ge=0, le=1, description="Headache present (1=Yes)", example=1)
    myalgia: int = Field(..., ge=0, le=1, description="Myalgia/body aches (1=Yes)", example=1)
    cough: int = Field(..., ge=0, le=1, description="Cough present (1=Yes)", example=0)
    nausea_vomiting: int = Field(..., ge=0, le=1, description="Nausea/vomiting (1=Yes)", example=1)
    abdominal_pain: int = Field(..., ge=0, le=1, description="Abdominal pain (1=Yes)", example=1)
    breathlessness: int = Field(..., ge=0, le=1, description="Breathlessness (1=Yes)", example=0)
    chills: int = Field(..., ge=0, le=1, description="Chills and rigors (1=Yes)", example=1)
    altered_sensorium: int = Field(..., ge=0, le=1, description="Altered sensorium/confusion (1=Yes)", example=0)
    jaundice: int = Field(..., ge=0, le=1, description="Jaundice present (1=Yes)", example=0)
    lymphadenopathy: int = Field(..., ge=0, le=1, description="Lymphadenopathy (1=Yes)", example=0)
    rash: int = Field(..., ge=0, le=1, description="Skin rash (1=Yes)", example=0)
    eschar: int = Field(..., ge=0, le=1, description="Eschar (pathognomonic black scab) present (1=Yes)", example=1)
    upper_eyelid_edema: int = Field(..., ge=0, le=1, description="Upper eyelid/facial edema (1=Yes)", example=1)

    # ── Laboratory Parameters ────────────────────────────────────────────
    wbc_count: float = Field(..., ge=500.0, le=100000.0, description="WBC count (cells/μL)", example=7800.0)
    platelet_count: float = Field(..., ge=5000.0, le=800000.0, description="Platelet count (cells/μL)", example=95000.0)
    hemoglobin: float = Field(..., ge=3.0, le=20.0, description="Hemoglobin (g/dL)", example=10.5)
    ast: float = Field(..., ge=5.0, le=5000.0, description="AST/SGOT (U/L)", example=120.0)
    alt: float = Field(..., ge=5.0, le=5000.0, description="ALT/SGPT (U/L)", example=95.0)
    blood_urea: float = Field(..., ge=5.0, le=500.0, description="Blood urea (mg/dL)", example=55.0)
    serum_creatinine: float = Field(..., ge=0.1, le=20.0, description="Serum creatinine (mg/dL)", example=1.1)
    crp: float = Field(..., ge=0.0, le=500.0, description="C-reactive protein (mg/L)", example=45.0)
    sodium: float = Field(..., ge=100.0, le=175.0, description="Serum sodium (mEq/L)", example=134.0)
    bilirubin: float = Field(..., ge=0.1, le=30.0, description="Total bilirubin (mg/dL)", example=0.9)

    # ── Diagnostics ──────────────────────────────────────────────────────
    elisa_igm: int = Field(..., ge=0, le=1, description="IgM ELISA result (1=Positive, 0=Negative/Not done)", example=1)
    weil_felix: int = Field(..., ge=0, le=1, description="Weil-Felix test (1=Positive, 0=Negative/Not done)", example=0)

    # ── Clinical Cross-Validators ────────────────────────────────────────
    @field_validator("fever_temp_c")
    @classmethod
    def validate_fever(cls, v: float) -> float:
        if v < 37.5:
            raise ValueError("fever_temp_c must be ≥ 37.5 °C for febrile illness workup")
        return v

    @field_validator("platelet_count")
    @classmethod
    def validate_platelets(cls, v: float) -> float:
        if v > 800000:
            raise ValueError("Platelet count >800,000 is clinically implausible; verify input")
        return v

    @model_validator(mode="after")
    def cross_validate_liver(self) -> "PatientInput":
        if self.jaundice == 1 and self.bilirubin < 2.0:
            raise ValueError(
                "Clinical inconsistency: jaundice reported but bilirubin < 2.0 mg/dL. "
                "Please verify laboratory result."
            )
        return self


class FeatureContribution(BaseModel):
    feature: str
    importance: float
    value: Any
    clinical_note: str


class TreatmentRecommendation(BaseModel):
    drug: str
    dose: str
    route: str
    duration: str
    indication: str
    priority: str


class PredictionResponse(BaseModel):
    request_id: str
    timestamp: str
    prediction: int                      # 0 or 1
    diagnosis: str                       # Human label
    probability_positive: float          # P(scrub typhus)
    probability_negative: float
    risk_level: str                      # LOW / MODERATE / HIGH / CRITICAL
    risk_score_pct: float                # 0–100 for UI gauge
    top_features: List[FeatureContribution]
    treatment_recommendations: List[TreatmentRecommendation]
    complications_to_watch: List[str]
    clinical_summary: str
    model_confidence: str
    disclaimer: str


class ModelInfoResponse(BaseModel):
    model_type: str
    accuracy: float
    roc_auc: float
    f1_score: float
    recall: float
    precision: float
    cv_roc_auc: str
    n_train: int
    n_test: int
    feature_count: int
    top_features: Dict[str, float]
    class_balance: Dict[str, int]
    confusion_matrix: List[List[int]]


# ─── Clinical Knowledge Engine ──────────────────────────────────────────────

CLINICAL_NOTES: Dict[str, str] = {
    "eschar": "Pathognomonic black eschar – hallmark of Orientia tsutsugamushi bite site",
    "fever_days": "Prolonged fever > 5 days significantly raises scrub typhus probability",
    "platelet_count": "Thrombocytopenia (<1.5 lakh) is a key lab marker in rickettsial disease",
    "ast": "AST elevation > 4× ULN indicates anicteric hepatitis – characteristic finding",
    "blood_urea": "Elevated BUN suggests early renal involvement (87% incidence per Pathania 2019)",
    "crp": "High CRP indicates systemic inflammatory response",
    "outdoor_exposure": "Outdoor/field exposure is the primary transmission risk factor",
    "monsoon_season": "Jul–Sep monsoon peaks mite activity and human exposure",
    "myalgia": "Myalgia (74% in Uttarakhand study) indicates systemic rickettsial involvement",
    "upper_eyelid_edema": "Upper eyelid edema (74% in Himalayan study) – distinctive regional finding",
    "rash": "Maculopapular rash – present in ~31% of Indian cases",
    "altered_sensorium": "Neurological involvement – indicates severe disease, requires ICU monitoring",
    "rural_background": "Rural residents have significantly higher mite exposure",
    "elisa_igm": "Positive IgM ELISA confirms active Orientia infection",
}


def get_treatment_recommendations(prob: float, data: PatientInput) -> List[TreatmentRecommendation]:
    recs = []
    is_pregnant = False  # could be extended

    if prob >= 0.4:
        # First-line
        recs.append(TreatmentRecommendation(
            drug="Doxycycline",
            dose="100 mg twice daily",
            route="Oral",
            duration="5–7 days (uncomplicated); 10–14 days (complicated)",
            indication="First-line treatment for scrub typhus in non-pregnant adults (WHO & IDSA)",
            priority="PRIMARY",
        ))
        # Alternative (intolerance / pregnancy)
        recs.append(TreatmentRecommendation(
            drug="Azithromycin",
            dose="500 mg once daily",
            route="Oral / IV",
            duration="5 days",
            indication="Alternative for doxycycline-intolerant patients; drug of choice in pregnancy",
            priority="ALTERNATIVE",
        ))
        if data.altered_sensorium or data.breathlessness:
            recs.append(TreatmentRecommendation(
                drug="IV Azithromycin + ICU Supportive Care",
                dose="500 mg OD IV",
                route="Intravenous",
                duration="5–7 days",
                indication="Severe/complicated disease with CNS or respiratory involvement",
                priority="CRITICAL",
            ))
        if data.platelet_count < 50000:
            recs.append(TreatmentRecommendation(
                drug="Platelet Transfusion (if <20,000 or active bleeding)",
                dose="As per hematology protocol",
                route="IV",
                duration="As needed",
                indication="Severe thrombocytopenia risk management",
                priority="SUPPORTIVE",
            ))
    return recs


def get_complications(data: PatientInput) -> List[str]:
    comps = []
    if data.altered_sensorium: comps.append("⚠️ Meningitis / Meningoencephalitis – Immediate neurology consult")
    if data.breathlessness: comps.append("⚠️ ARDS / Pneumonia – Chest X-ray, SpO2 monitoring, consider ICU")
    if data.platelet_count < 100000: comps.append("⚠️ Thrombocytopenia – Monitor for bleeding, repeat CBC daily")
    if data.serum_creatinine > 1.5: comps.append("⚠️ Acute Kidney Injury (AKI) – Monitor urine output, renal function tests")
    if data.ast > 120: comps.append("⚠️ Acute Hepatitis – Monitor LFTs; bilirubin for hepatic encephalopathy")
    if data.blood_urea > 60: comps.append("⚠️ Renal impairment – Monitor blood urea, creatinine daily")
    if data.jaundice: comps.append("⚠️ Hepatic encephalopathy risk – Monitor mental status")
    if data.fever_days > 10: comps.append("⚠️ Prolonged fever > 10 days – Check for treatment failure / co-infection")
    if not comps:
        comps.append("✅ No immediate high-risk complications identified. Monitor vitals and repeat labs in 48h.")
    return comps


def get_risk_level(prob: float) -> tuple[str, float]:
    score = prob * 100
    if prob < 0.25:
        return "LOW", score
    elif prob < 0.50:
        return "MODERATE", score
    elif prob < 0.75:
        return "HIGH", score
    else:
        return "CRITICAL", score


def build_feature_contributions(input_data: pd.DataFrame, data: PatientInput) -> List[FeatureContribution]:
    contribs = []
    for feat, imp in list(FEATURE_IMPORTANCE.items())[:12]:
        val = input_data[feat].values[0]
        note = CLINICAL_NOTES.get(feat, "Epidemiologically relevant clinical variable")
        contribs.append(FeatureContribution(
            feature=feat.replace("_", " ").title(),
            importance=round(float(imp), 4),
            value=round(float(val), 2) if isinstance(val, float) else int(val),
            clinical_note=note,
        ))
    return contribs


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Scrub Typhus Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/api/docs",
        "health": "/health",
    }


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": PIPELINE is not None,
        "model_accuracy": MODEL_METRICS["accuracy"],
    }


@app.get("/api/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    top10 = dict(list(FEATURE_IMPORTANCE.items())[:10])
    cb = {str(k): v for k, v in MODEL_METRICS["class_balance"].items()}
    return ModelInfoResponse(
        model_type="Random Forest Classifier (sklearn Pipeline + StandardScaler)",
        accuracy=MODEL_METRICS["accuracy"],
        roc_auc=MODEL_METRICS["roc_auc"],
        f1_score=MODEL_METRICS["f1_score"],
        recall=MODEL_METRICS["recall"],
        precision=MODEL_METRICS["precision"],
        cv_roc_auc=f"{MODEL_METRICS['cv_roc_auc_mean']:.4f} ± {MODEL_METRICS['cv_roc_auc_std']:.4f}",
        n_train=MODEL_METRICS["n_train"],
        n_test=MODEL_METRICS["n_test"],
        feature_count=MODEL_METRICS["n_features"],
        top_features=top10,
        class_balance=cb,
        confusion_matrix=MODEL_METRICS["confusion_matrix"],
    )


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientInput):
    req_id = str(uuid.uuid4())[:12]
    logger.info("[%s] Received prediction request", req_id)

    try:
        # Build DataFrame in correct feature order
        input_dict = patient.model_dump()
        input_df = pd.DataFrame([input_dict])[FEATURE_NAMES]

        # Predict
        prob_arr = PIPELINE.predict_proba(input_df)[0]
        prob_negative, prob_positive = float(prob_arr[0]), float(prob_arr[1])
        prediction = int(prob_positive >= 0.5)

        risk_level, risk_score = get_risk_level(prob_positive)
        diagnosis = "Scrub Typhus – Positive" if prediction == 1 else "Scrub Typhus – Negative"

        top_features = build_feature_contributions(input_df, patient)
        treatment = get_treatment_recommendations(prob_positive, patient)
        complications = get_complications(patient)

        # Clinical summary
        if prediction == 1:
            summary = (
                f"High index of suspicion for Scrub Typhus (Orientia tsutsugamushi). "
                f"Probability: {prob_positive*100:.1f}%. Risk: {risk_level}. "
                f"Patient presents with {patient.fever_days}-day fever"
                + (", eschar" if patient.eschar else "")
                + (", thrombocytopenia" if patient.platelet_count < 150000 else "")
                + (", elevated transaminases" if patient.ast > 80 else "")
                + ". Initiate empirical doxycycline without waiting for serology confirmation."
            )
        else:
            summary = (
                f"Low probability of Scrub Typhus ({prob_positive*100:.1f}%). "
                f"Risk: {risk_level}. Consider alternative diagnoses: malaria, dengue, leptospirosis, "
                f"enteric fever. Re-evaluate if fever persists > 5 days with outdoor exposure history."
            )

        confidence = (
            "Very High (>90%)" if abs(prob_positive - 0.5) > 0.4
            else "High (75–90%)" if abs(prob_positive - 0.5) > 0.25
            else "Moderate (50–75%)"
        )

        return PredictionResponse(
            request_id=req_id,
            timestamp=datetime.utcnow().isoformat(),
            prediction=prediction,
            diagnosis=diagnosis,
            probability_positive=round(prob_positive, 4),
            probability_negative=round(prob_negative, 4),
            risk_level=risk_level,
            risk_score_pct=round(risk_score, 1),
            top_features=top_features,
            treatment_recommendations=treatment,
            complications_to_watch=complications,
            clinical_summary=summary,
            model_confidence=confidence,
            disclaimer=(
                "This AI tool is for clinical decision support only. "
                "It does not replace physician judgment. "
                "Diagnosis should be confirmed with IgM ELISA serology. "
                "Initiate treatment empirically in endemic areas if clinical suspicion is high."
            ),
        )

    except Exception as exc:
        logger.error("[%s] Prediction error: %s", req_id, exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc))


@app.get("/api/dataset-stats", tags=["Data"])
def dataset_stats():
    """Return summary statistics of training dataset for dashboard."""
    try:
        df = pd.read_csv(DATA_DIR / "scrub_typhus_20k.csv")
        stats = {
            "total_records": len(df),
            "positive_cases": int(df["scrub_typhus"].sum()),
            "negative_cases": int((df["scrub_typhus"] == 0).sum()),
            "prevalence_pct": round(df["scrub_typhus"].mean() * 100, 2),
            "mean_age": round(df["age"].mean(), 1),
            "eschar_prevalence_pct": round(df["eschar"].mean() * 100, 1),
            "mean_fever_days": round(df["fever_days"].mean(), 1),
            "mean_platelet": round(df["platelet_count"].mean(), 0),
            "mean_ast": round(df["ast"].mean(), 1),
            "monsoon_cases_pct": round(df[df["monsoon_season"] == 1]["scrub_typhus"].mean() * 100, 1),
            "rural_cases_pct": round(df[df["rural_background"] == 1]["scrub_typhus"].mean() * 100, 1),
        }
        return stats
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
