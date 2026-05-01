"""
ChurnGuard AI — Feature Engineering Module
Adds business-meaningful derived features.
"""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the customer dataframe."""
    df = df.copy()

    # Engagement & Usage
    df["engagement_rate"] = (df["active_days"] / 30.0).clip(0, 1)
    df["usage_per_login"] = df["monthly_usage_hours"] / (df["login_count"] + 1e-3)
    df["session_depth"] = df["avg_session_min"] * df["login_count"]

    # Support & Risk
    df["support_intensity"] = df["support_tickets"] + 3 * df["sla_breaches"]
    df["ticket_to_tenure"] = df["support_tickets"] / (df["tenure_months"] + 1e-3)

    # Email Engagement
    df["email_ctr"] = df["email_clicks"] / (df["email_opens"] + 1e-3)
    df["email_engagement_score"] = (df["email_opens"] * 0.3 + df["email_clicks"] * 0.7).clip(0)

    # Price & Value
    df["price_to_tenure"] = df["billing_amount"] / (df["tenure_months"] + 1e-3)
    df["value_score"] = df["monthly_usage_hours"] / (df["billing_amount"] + 1e-3)
    df["discount_leverage"] = (df["is_discounted"].astype(float) * df["billing_amount"])

    # Recency
    df["payment_recency_flag"] = (df["last_payment_days_ago"] > 30).astype(int)
    df["campaign_recency_flag"] = (df["last_campaign_days_ago"] > 60).astype(int)

    # Loyalty signals
    df["is_long_tenure"] = (df["tenure_months"] > 24).astype(int)
    df["multi_device"] = (df["device_count"] > 2).astype(int)
    df["add_on_density"] = df["add_on_count"] / (df["device_count"] + 1e-3)

    # Boolean -> int columns for the numeric pipeline (avoids dtype errors in sklearn)
    df["is_autopay_int"] = df["is_autopay"].astype(int)
    df["is_discounted_int"] = df["is_discounted"].astype(int)
    df["has_family_bundle_int"] = df["has_family_bundle"].astype(int)

    return df
