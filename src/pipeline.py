"""
ChurnGuard AI — Preprocessing Pipeline
ColumnTransformer for numerical and categorical features.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Numerical features (including engineered)
NUM_FEATURES = [
    "billing_amount", "last_payment_days_ago", "tenure_months",
    "monthly_usage_hours", "active_days", "login_count", "avg_session_min",
    "device_count", "add_on_count", "support_tickets", "sla_breaches",
    "promotions_redeemed", "email_opens", "email_clicks",
    "last_campaign_days_ago", "nps_score",
    # Engineered
    "engagement_rate", "usage_per_login", "session_depth",
    "support_intensity", "ticket_to_tenure",
    "email_ctr", "email_engagement_score",
    "price_to_tenure", "value_score", "discount_leverage",
    "payment_recency_flag", "campaign_recency_flag",
    "is_long_tenure", "multi_device", "add_on_density",
    # Boolean flags (cast to int in add_features)
    "is_autopay_int", "is_discounted_int", "has_family_bundle_int",
]

# String categorical features only
CAT_FEATURES = ["plan_tier", "region"]

# Pipelines
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# Full preprocessor
preprocessor = ColumnTransformer([
    ("num", num_pipe, NUM_FEATURES),
    ("cat", cat_pipe, CAT_FEATURES),
], remainder="drop")
