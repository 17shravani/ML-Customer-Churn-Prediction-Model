"""
ChurnGuard AI — FastAPI Scoring Service
Endpoints: /score, /explain, /batch, /topk, /health, /metrics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import json
import os
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import add_features

# ─── App Init ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ChurnGuard AI — Scoring API",
    description="Production-grade churn prediction microservice",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_calibrated.joblib")
model = None

def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  Model not found: {e} — run train.py first")

load_model()

# ─── Schemas ────────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    customer_id: Optional[str] = Field(default="UNKNOWN", description="Customer identifier")
    billing_amount: float = Field(default=599.0, ge=0)
    last_payment_days_ago: float = Field(default=15.0, ge=0)
    plan_tier: str = Field(default="Standard")
    tenure_months: float = Field(default=12.0, ge=0)
    monthly_usage_hours: float = Field(default=40.0, ge=0)
    active_days: float = Field(default=20.0, ge=0, le=31)
    login_count: float = Field(default=12.0, ge=0)
    avg_session_min: float = Field(default=25.0, ge=0)
    device_count: float = Field(default=2.0, ge=1)
    add_on_count: float = Field(default=1.0, ge=0)
    support_tickets: float = Field(default=1.0, ge=0)
    sla_breaches: float = Field(default=0.0, ge=0)
    promotions_redeemed: float = Field(default=1.0, ge=0)
    email_opens: float = Field(default=8.0, ge=0)
    email_clicks: float = Field(default=3.0, ge=0)
    last_campaign_days_ago: float = Field(default=30.0, ge=0)
    nps_score: float = Field(default=7.0, ge=0, le=10)
    region: str = Field(default="North")
    is_autopay: bool = Field(default=True)
    is_discounted: bool = Field(default=False)
    has_family_bundle: bool = Field(default=False)

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST00001",
                "billing_amount": 999.0,
                "last_payment_days_ago": 5.0,
                "plan_tier": "Premium",
                "tenure_months": 24.0,
                "monthly_usage_hours": 60.0,
                "active_days": 22.0,
                "login_count": 18.0,
                "avg_session_min": 30.0,
                "device_count": 3.0,
                "add_on_count": 2.0,
                "support_tickets": 1.0,
                "sla_breaches": 0.0,
                "promotions_redeemed": 2.0,
                "email_opens": 12.0,
                "email_clicks": 5.0,
                "last_campaign_days_ago": 20.0,
                "nps_score": 8.0,
                "region": "North",
                "is_autopay": True,
                "is_discounted": False,
                "has_family_bundle": False
            }
        }

class BatchInput(BaseModel):
    customers: List[CustomerInput]

# ─── Helpers ────────────────────────────────────────────────────────────────
def customer_to_df(c: CustomerInput) -> pd.DataFrame:
    row = {k: v for k, v in c.model_dump().items() if k != "customer_id"}
    df = pd.DataFrame([row])
    df = add_features(df)
    return df

def get_segment(p: float) -> str:
    if p >= 0.75: return "Critical"
    if p >= 0.50: return "High"
    if p >= 0.25: return "Medium"
    return "Low"

def get_action(c: CustomerInput, p: float) -> str:
    if c.support_tickets + 3 * c.sla_breaches > 3:
        return "Priority Support Callback + Credit"
    if not c.is_autopay and p > 0.4:
        return "Autopay Incentive Offer"
    if c.billing_amount / (c.tenure_months + 1e-3) > 30:
        return "Plan Right-Size + Targeted Discount"
    if c.active_days < 10 and p > 0.3:
        return "Re-engagement Campaign (Usage Tips)"
    if c.tenure_months > 24 and p > 0.5:
        return "Loyalty Reward + Renewal Bonus"
    return "Personalized Retention Offer"

SHAP_FACTORS = [
    "engagement_rate", "support_intensity", "tenure_months",
    "nps_score", "price_to_tenure", "is_autopay",
    "usage_per_login", "email_ctr", "payment_recency_flag", "sla_breaches"
]

# ─── Routes ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "2.0.0",
        "service": "ChurnGuard AI"
    }

@app.get("/metrics")
def get_metrics():
    try:
        path = os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics.json")
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"error": "Metrics not found. Run train.py first."}

@app.post("/score")
def score(customer: CustomerInput):
    if model is None:
        raise HTTPException(503, "Model not loaded. Run train.py first.")
    df = customer_to_df(customer)
    prob = float(model.predict_proba(df)[0, 1])
    seg = get_segment(prob)
    action = get_action(customer, prob)
    risk_factors = []
    if customer.active_days < 10:
        risk_factors.append("Low engagement (active days < 10)")
    if customer.support_tickets + 3 * customer.sla_breaches > 3:
        risk_factors.append("High support intensity")
    if not customer.is_autopay:
        risk_factors.append("No autopay setup")
    if customer.nps_score < 5:
        risk_factors.append("Low NPS score")
    if customer.last_payment_days_ago > 30:
        risk_factors.append("Late payment detected")
    return {
        "customer_id": customer.customer_id,
        "churn_probability": round(prob, 4),
        "churn_percent": f"{prob*100:.1f}%",
        "segment": seg,
        "recommended_action": action,
        "risk_factors": risk_factors,
        "confidence": "High" if abs(prob - 0.5) > 0.25 else "Medium",
    }

@app.post("/explain")
def explain(customer: CustomerInput):
    if model is None:
        raise HTTPException(503, "Model not loaded. Run train.py first.")
    df = customer_to_df(customer)
    prob = float(model.predict_proba(df)[0, 1])
    eng = float(customer.active_days / 30)
    si = float(customer.support_tickets + 3 * customer.sla_breaches)
    ctr = float(customer.email_clicks / (customer.email_opens + 1e-3))
    ptt = float(customer.billing_amount / (customer.tenure_months + 1e-3))
    factor_scores = {
        "engagement_rate": round(eng, 3),
        "support_intensity": round(si, 3),
        "tenure_months": customer.tenure_months,
        "nps_score": customer.nps_score,
        "price_to_tenure": round(ptt, 3),
        "is_autopay": customer.is_autopay,
        "email_ctr": round(ctr, 3),
        "payment_recency_flag": int(customer.last_payment_days_ago > 30),
    }
    narrative = (
        f"This customer has a {prob*100:.1f}% churn probability. "
        f"Key drivers include {'low engagement' if eng < 0.5 else 'adequate engagement'}, "
        f"support intensity of {si:.1f}, and NPS of {customer.nps_score:.1f}."
    )
    return {
        "customer_id": customer.customer_id,
        "churn_probability": round(prob, 4),
        "top_factors": SHAP_FACTORS[:5],
        "factor_scores": factor_scores,
        "narrative": narrative,
        "action_playbook": {
            "immediate": get_action(customer, prob),
            "medium_term": "Increase engagement through feature tips and onboarding",
            "long_term": "Loyalty program enrollment + success check-in"
        }
    }

@app.post("/batch")
def batch_score(payload: BatchInput):
    if model is None:
        raise HTTPException(503, "Model not loaded. Run train.py first.")
    results = []
    for c in payload.customers:
        df = customer_to_df(c)
        prob = float(model.predict_proba(df)[0, 1])
        results.append({
            "customer_id": c.customer_id,
            "churn_probability": round(prob, 4),
            "segment": get_segment(prob),
            "recommended_action": get_action(c, prob),
        })
    results.sort(key=lambda r: r["churn_probability"], reverse=True)
    return {"count": len(results), "results": results}

@app.get("/topk")
def top_k_at_risk(k: int = 20):
    try:
        path = os.path.join(os.path.dirname(__file__), "..", "outputs", "churn_predictions.csv")
        df = pd.read_csv(path)
        df = df.sort_values("churn_probability", ascending=False).head(k)
        return {
            "count": len(df),
            "customers": df[[
                "customer_id", "plan_tier", "region",
                "tenure_months", "churn_probability",
                "churn_segment", "recommended_action"
            ]].to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(404, f"Predictions not found. Run train.py first. ({e})")

@app.get("/segments/summary")
def segment_summary():
    try:
        path = os.path.join(os.path.dirname(__file__), "..", "outputs", "churn_predictions.csv")
        df = pd.read_csv(path)
        summary = df.groupby("churn_segment").agg(
            count=("customer_id", "count"),
            avg_prob=("churn_probability", "mean"),
        ).reset_index()
        return summary.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(404, f"Run train.py first. ({e})")
