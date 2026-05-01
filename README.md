# 🛡️ ChurnGuard AI — Customer Churn Prediction Model

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/XGBoost-Calibrated-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FastAPI-Scoring%20API-green?style=for-the-badge&logo=fastapi"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit"/>
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-purple?style=for-the-badge"/>
</p>

> **Industry-grade ML system** that predicts customer churn, explains key risk drivers using SHAP, and exposes a FastAPI scoring API — all visualised in a premium Streamlit dashboard with dark/light mode.

---

## 🎯 Problem Statement

Customer churn costs businesses billions annually. A **5% reduction in churn** can increase profits by **25–95%** (Harvard Business Review). ChurnGuard AI uses machine learning to:
- Predict which customers will churn next billing cycle
- Explain *why* they are at risk (SHAP explainability)
- Recommend targeted retention actions
- Expose real-time scoring via REST API

---

## 🏗️ Architecture

```
Synthetic Data Generator
        ↓
Feature Engineering (30+ features)
        ↓
Preprocessing Pipeline (ColumnTransformer)
        ↓
XGBoost → Isotonic Calibration
        ↓
  ┌─────────────────────────┐
  │   FastAPI Scoring API   │  ← /score  /explain  /batch  /topk
  └─────────────────────────┘
              ↓
  ┌─────────────────────────┐
  │   Streamlit Dashboard   │  ← 5 pages, dark/light theme
  └─────────────────────────┘
```

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 **ML Model** | XGBoost + Isotonic Calibration |
| 📊 **Metrics** | ROC-AUC, PR-AUC, Lift@10%, Brier Score |
| 🔍 **Explainability** | SHAP feature importance |
| ⚡ **REST API** | FastAPI with `/score`, `/explain`, `/batch`, `/topk` |
| 🎨 **Dashboard** | Streamlit — 5 pages, dark/light theme, Plotly charts |
| 🐳 **Docker** | Production-ready containerisation |
| 📦 **Segments** | Low / Medium / High / Critical risk tiers |
| 💡 **Actions** | Automated retention recommendation engine |


├── Dockerfile
└── README.md
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health check |
| GET | `/metrics` | Model performance metrics |
| POST | `/score` | Score a single customer |
| POST | `/explain` | Get SHAP-style explanation |
| POST | `/batch` | Batch score multiple customers |
| GET | `/topk?k=20` | Top-K at-risk customers |
| GET | `/segments/summary` | Segment distribution |

**Example `/score` request:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"CUST001","billing_amount":999,"tenure_months":6,
       "active_days":8,"login_count":3,"support_tickets":4,"sla_breaches":2,
       "nps_score":3.5,"is_autopay":false,"plan_tier":"Premium",
       "region":"North","monthly_usage_hours":10,"avg_session_min":12,
       "device_count":1,"add_on_count":0,"promotions_redeemed":0,
       "email_opens":2,"email_clicks":0,"last_campaign_days_ago":90,
       "last_payment_days_ago":35,"is_discounted":false,"has_family_bundle":false}'
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | ~0.89 |
| PR-AUC | ~0.72 |
| Lift @ 10% | ~3.5x |
| Baseline Churn Rate | ~22% |

---

## 🧠 Key Features Engineered

| Feature | Business Meaning |
|---|---|
| `engagement_rate` | Active days / 30 — usage intensity |
| `support_intensity` | Tickets + 3×SLA breaches — frustration proxy |
| `price_to_tenure` | Billing / tenure — perceived value |
| `email_ctr` | Click-through rate — campaign receptiveness |
| `payment_recency_flag` | Late payment signal |

---

## 💼 Resume Bullet Points

- Built an **end-to-end churn prediction system** achieving **ROC-AUC 0.89** and **3.5× Lift@10%** using XGBoost + isotonic calibration on 5,000 synthetic customers
- Engineered **30+ business features** (engagement rate, support intensity, price-to-tenure) from raw billing, usage, and support data
- Deployed a **FastAPI microservice** with `/score`, `/explain`, `/batch` endpoints and a **Streamlit analytics dashboard** with dark/light theme
- Implemented **SHAP explainability** to surface top churn drivers and generate automated retention action recommendations

---

## 🐳 Docker Deployment

```bash
docker build -t churnguard-api .
docker run -p 8000:8000 churnguard-api


## 🌍 Real-World Applications

- **Telecom**: Predict SIM churn, trigger win-back offers
- **SaaS**: Success-ops triage queue for CSMs
- **OTT/Streaming**: Lapsed viewer re-engagement
- **Fintech**: Loan app inactivity / account closure

