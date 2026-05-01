"""
ChurnGuard AI — Synthetic Data Generator
Generates a realistic customer churn dataset for telecom/SaaS simulation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

N = 5000  # number of customers

def generate_churn_data(n=N):
    regions = ["North", "South", "East", "West", "Central"]
    plan_tiers = ["Basic", "Standard", "Premium", "Enterprise"]
    
    # Base features
    tenure_months = np.random.exponential(scale=24, size=n).clip(1, 120).astype(int)
    plan_tier = np.random.choice(plan_tiers, n, p=[0.30, 0.35, 0.25, 0.10])
    billing_map = {"Basic": 299, "Standard": 599, "Premium": 999, "Enterprise": 1999}
    billing_amount = np.array([billing_map[p] for p in plan_tier]) * np.random.uniform(0.85, 1.15, n)

    region = np.random.choice(regions, n)
    is_autopay = np.random.binomial(1, 0.55, n).astype(bool)
    is_discounted = np.random.binomial(1, 0.30, n).astype(bool)
    has_family_bundle = np.random.binomial(1, 0.20, n).astype(bool)
    nps_score = np.random.normal(6.5, 2.5, n).clip(0, 10)

    monthly_usage_hours = np.random.exponential(40, n).clip(0, 200)
    active_days = np.random.randint(0, 31, n)
    login_count = np.random.poisson(15, n)
    avg_session_min = np.random.exponential(25, n).clip(1, 120)
    device_count = np.random.randint(1, 6, n)
    add_on_count = np.random.randint(0, 5, n)
    support_tickets = np.random.poisson(1.5, n)
    sla_breaches = np.random.poisson(0.3, n)
    promotions_redeemed = np.random.randint(0, 4, n)
    email_opens = np.random.randint(0, 30, n)
    email_clicks = np.clip(np.random.randint(0, email_opens + 1, n), 0, email_opens)
    last_payment_days_ago = np.random.randint(1, 45, n)
    last_campaign_days_ago = np.random.randint(1, 120, n)

    # Cycle dates
    base_date = datetime(2024, 1, 1)
    cycle_starts = [base_date + timedelta(days=random.randint(0, 300)) for _ in range(n)]
    cycle_ends = [s + timedelta(days=30) for s in cycle_starts]

    # Feature engineering for churn probability
    engagement_rate = (active_days / 30.0).clip(0, 1)
    usage_per_login = monthly_usage_hours / (login_count + 1e-3)
    support_intensity = support_tickets + 3 * sla_breaches
    email_ctr = email_clicks / (email_opens + 1e-3)
    price_to_tenure = billing_amount / (tenure_months + 1e-3)

    # Churn probability formula (business logic)
    churn_logit = (
        -2.0
        - 0.05 * tenure_months
        - 1.5 * engagement_rate
        + 0.4 * support_intensity
        - 0.3 * nps_score
        + 0.002 * price_to_tenure
        - 0.8 * is_autopay.astype(float)
        - 0.5 * has_family_bundle.astype(float)
        + 0.003 * last_payment_days_ago
        - 0.3 * email_ctr
        + np.random.normal(0, 0.5, n)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churned_next_cycle = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST{str(i).zfill(5)}" for i in range(n)],
        "cycle_start": [s.strftime("%Y-%m-%d") for s in cycle_starts],
        "cycle_end": [e.strftime("%Y-%m-%d") for e in cycle_ends],
        "billing_amount": billing_amount.round(2),
        "last_payment_days_ago": last_payment_days_ago.astype(float),
        "plan_tier": plan_tier,
        "tenure_months": tenure_months.astype(float),
        "monthly_usage_hours": monthly_usage_hours.round(2),
        "active_days": active_days.astype(float),
        "login_count": login_count.astype(float),
        "avg_session_min": avg_session_min.round(2),
        "device_count": device_count.astype(float),
        "add_on_count": add_on_count.astype(float),
        "support_tickets": support_tickets.astype(float),
        "sla_breaches": sla_breaches.astype(float),
        "promotions_redeemed": promotions_redeemed.astype(float),
        "email_opens": email_opens.astype(float),
        "email_clicks": email_clicks.astype(float),
        "last_campaign_days_ago": last_campaign_days_ago.astype(float),
        "nps_score": nps_score.round(2),
        "region": region,
        "is_autopay": is_autopay,
        "is_discounted": is_discounted,
        "has_family_bundle": has_family_bundle,
        "churned_next_cycle": churned_next_cycle,
    })

    print(f"✅ Generated {n} customer records")
    print(f"📊 Churn Rate: {churned_next_cycle.mean():.2%}")
    print(f"📁 Shape: {df.shape}")
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_churn_data()
    df.to_csv("data/churn_frame.csv", index=False)
    df.to_parquet("data/churn_frame.parquet", index=False)
    print("✅ Saved: data/churn_frame.csv & data/churn_frame.parquet")
