"""
ChurnGuard AI — Streamlit Dashboard
Premium dark/light themed analytics dashboard.
"""
import os, sys, json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

sys.path.insert(0, os.path.dirname(__file__))

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ──────────────────────────────────────────────────────
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0F0F1A; color: #E8E8F0; }
.main .block-container { padding: 1.5rem 2rem; }
[data-testid="stSidebar"] { background: #16162A !important; border-right: 1px solid #2D2D4E; }
.metric-card {
    background: linear-gradient(135deg, #1E1E35, #252545);
    border: 1px solid #3D3D6B;
    border-radius: 16px; padding: 1.2rem 1.5rem;
    box-shadow: 0 4px 20px rgba(108,99,255,0.15);
    transition: transform 0.2s; cursor: default;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-val { font-size: 2.2rem; font-weight: 800; color: #6C63FF; }
.metric-lbl { font-size: 0.82rem; color: #9090B8; margin-top: 4px; }
.seg-critical { background: linear-gradient(90deg,#FF4444,#FF6B6B22); border-left: 4px solid #FF4444; border-radius: 8px; padding: 8px 14px; }
.seg-high { background: linear-gradient(90deg,#FF8C00,#FF8C0022); border-left: 4px solid #FF8C00; border-radius: 8px; padding: 8px 14px; }
.seg-medium { background: linear-gradient(90deg,#FFD700,#FFD70022); border-left: 4px solid #FFD700; border-radius: 8px; padding: 8px 14px; }
.seg-low { background: linear-gradient(90deg,#43E97B,#43E97B22); border-left: 4px solid #43E97B; border-radius: 8px; padding: 8px 14px; }
div[data-testid="stButton"] button {
    background: linear-gradient(135deg,#6C63FF,#9F97FF);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}
div[data-testid="stButton"] button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(108,99,255,0.4); }
h1,h2,h3 { color: #E8E8F0 !important; }
.stTabs [data-baseweb="tab"] { color: #9090B8 !important; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #6C63FF !important; border-bottom: 2px solid #6C63FF !important; }
</style>
"""

LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background: #F4F4FF; color: #1A1A2E; }
.main .block-container { padding: 1.5rem 2rem; }
[data-testid="stSidebar"] { background: #EEEEFF !important; }
.metric-card {
    background: white; border: 1px solid #DDDDF5;
    border-radius: 16px; padding: 1.2rem 1.5rem;
    box-shadow: 0 4px 20px rgba(108,99,255,0.08);
}
.metric-val { font-size: 2.2rem; font-weight: 800; color: #5B52EE; }
.metric-lbl { font-size: 0.82rem; color: #6666AA; margin-top: 4px; }
</style>
"""

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
    st.title("ChurnGuard AI")
    st.caption("v2.0 — Retention Intelligence")
    st.divider()
    theme = st.radio("🎨 Theme", ["Dark", "Light"], index=0, horizontal=True)
    st.divider()
    page = st.selectbox("📂 Navigate", [
        "🏠 Overview Dashboard",
        "🔍 Score Customer",
        "📋 At-Risk Watchlist",
        "📊 Analytics & EDA",
        "🤖 Model Performance",
    ])
    st.divider()
    API = st.text_input("API URL", value="http://localhost:8000")
    st.caption("© 2025 ChurnGuard AI")

st.markdown(DARK_CSS if theme == "Dark" else LIGHT_CSS, unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "outputs/churn_predictions.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_metrics():
    path = "outputs/metrics.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

df = load_data()
metrics = load_metrics()

# ── Segment colors ─────────────────────────────────────────────
SEG_COLOR = {"Critical": "#FF4444", "High": "#FF8C00", "Medium": "#FFD700", "Low": "#43E97B"}

# ══════════════════════════════════════════════════════════════
# PAGE: Overview Dashboard
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview Dashboard":
    st.markdown("## 🛡️ ChurnGuard AI — Command Center")
    st.caption("Real-time customer churn intelligence & retention analytics")

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Total Customers", f"{len(df):,}" if df is not None else "—", "👥"),
        ("Churn Rate", f"{metrics.get('churn_rate',0)*100:.1f}%" if metrics else "—", "📉"),
        ("ROC-AUC", f"{metrics.get('roc_auc','—')}", "🎯"),
        ("PR-AUC", f"{metrics.get('pr_auc','—')}", "📈"),
        ("Lift@10%", f"{metrics.get('lift_at_10pct','—')}x", "⚡"),
    ]
    for col, (lbl, val, icon) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.4rem">{icon}</div>
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if df is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            seg_counts = df["churn_segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            fig = px.pie(seg_counts, values="Count", names="Segment",
                         title="Customer Risk Segments",
                         color="Segment", hole=0.55,
                         color_discrete_map=SEG_COLOR)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#E8E8F0", title_font_size=15)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.histogram(df, x="churn_probability", nbins=40,
                                title="Churn Score Distribution",
                                color_discrete_sequence=["#6C63FF"])
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#E8E8F0", title_font_size=15,
                               bargap=0.05)
            st.plotly_chart(fig2, use_container_width=True)

        # Region breakdown
        region_df = df.groupby("region")["churn_probability"].mean().reset_index()
        region_df.columns = ["Region", "Avg Churn Prob"]
        fig3 = px.bar(region_df.sort_values("Avg Churn Prob", ascending=True),
                      x="Avg Churn Prob", y="Region", orientation="h",
                      title="Avg Churn Probability by Region",
                      color="Avg Churn Prob", color_continuous_scale="Purples")
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#E8E8F0", title_font_size=15)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("⚠️ Run `python train.py` first to generate predictions.")

# ══════════════════════════════════════════════════════════════
# PAGE: Score Customer
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Score Customer":
    st.markdown("## 🔍 Real-Time Customer Churn Scorer")
    st.caption("Enter customer details to get instant churn risk and retention recommendation.")

    with st.form("score_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            cid = st.text_input("Customer ID", "CUST00001")
            plan = st.selectbox("Plan Tier", ["Basic","Standard","Premium","Enterprise"])
            region = st.selectbox("Region", ["North","South","East","West","Central"])
            tenure = st.slider("Tenure (months)", 1, 120, 12)
            billing = st.number_input("Billing Amount (₹)", 100.0, 5000.0, 599.0)
        with c2:
            active_days = st.slider("Active Days (last 30)", 0, 31, 20)
            login_count = st.slider("Login Count", 0, 60, 12)
            usage = st.slider("Monthly Usage (hrs)", 0.0, 200.0, 40.0)
            avg_session = st.slider("Avg Session (min)", 1.0, 120.0, 25.0)
            device_count = st.slider("Device Count", 1, 6, 2)
        with c3:
            support = st.slider("Support Tickets", 0, 10, 1)
            sla = st.slider("SLA Breaches", 0, 5, 0)
            nps = st.slider("NPS Score", 0.0, 10.0, 7.0)
            last_pay = st.slider("Last Payment (days ago)", 1, 45, 10)
            is_autopay = st.checkbox("Autopay Enabled", True)
            is_disc = st.checkbox("Discounted Plan", False)
            family = st.checkbox("Family Bundle", False)
        submitted = st.form_submit_button("🚀 Predict Churn Risk")

    if submitted:
        payload = {
            "customer_id": cid, "billing_amount": billing, "last_payment_days_ago": last_pay,
            "plan_tier": plan, "tenure_months": float(tenure), "monthly_usage_hours": usage,
            "active_days": float(active_days), "login_count": float(login_count),
            "avg_session_min": avg_session, "device_count": float(device_count),
            "add_on_count": 1.0, "support_tickets": float(support), "sla_breaches": float(sla),
            "promotions_redeemed": 1.0, "email_opens": 8.0, "email_clicks": 3.0,
            "last_campaign_days_ago": 30.0, "nps_score": nps, "region": region,
            "is_autopay": is_autopay, "is_discounted": is_disc, "has_family_bundle": family
        }
        try:
            res = requests.post(f"{API}/score", json=payload, timeout=5)
            data = res.json()
            prob = data["churn_probability"]
            seg = data["segment"]
            seg_colors = {"Critical":"#FF4444","High":"#FF8C00","Medium":"#FFD700","Low":"#43E97B"}
            color = seg_colors.get(seg, "#6C63FF")

            st.markdown("<br>", unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            r1.metric("Churn Probability", data["churn_percent"])
            r2.metric("Risk Segment", seg)
            r3.metric("Confidence", data["confidence"])

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100,
                title={"text": "Churn Risk Score", "font": {"color": "#E8E8F0"}},
                gauge={"axis": {"range": [0,100]},
                       "bar": {"color": color},
                       "steps": [{"range":[0,25],"color":"#43E97B22"},
                                 {"range":[25,50],"color":"#FFD70022"},
                                 {"range":[50,75],"color":"#FF8C0022"},
                                 {"range":[75,100],"color":"#FF444422"}]},
                number={"suffix":"%", "font":{"color":color,"size":36}}
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E8E8F0", height=300)
            st.plotly_chart(fig, use_container_width=True)

            st.success(f"💡 **Recommended Action:** {data['recommended_action']}")
            if data.get("risk_factors"):
                st.warning("⚠️ **Risk Factors Detected:**\n" + "\n".join([f"- {r}" for r in data["risk_factors"]]))
        except Exception as e:
            st.error(f"API Error: {e}. Make sure FastAPI is running on {API}")

# ══════════════════════════════════════════════════════════════
# PAGE: At-Risk Watchlist
# ══════════════════════════════════════════════════════════════
elif page == "📋 At-Risk Watchlist":
    st.markdown("## 📋 At-Risk Customer Watchlist")
    if df is not None:
        top_k = st.slider("Top K at-risk customers", 10, 100, 20)
        seg_filter = st.multiselect("Filter by Segment", ["Critical","High","Medium","Low"],
                                    default=["Critical","High"])
        filtered = df[df["churn_segment"].isin(seg_filter)].sort_values(
            "churn_probability", ascending=False).head(top_k)

        st.dataframe(
            filtered[["customer_id","plan_tier","region","tenure_months",
                       "churn_probability","churn_segment","recommended_action"]]
            .style.format({"churn_probability": "{:.1%}",
                           "tenure_months": "{:.0f} mo"})
            .map(lambda v: f"color: {SEG_COLOR.get(v,'#E8E8F0')}",
                      subset=["churn_segment"]),
            use_container_width=True, height=500
        )
        csv = filtered.to_csv(index=False).encode()
        st.download_button("⬇️ Download Watchlist CSV", csv, "watchlist.csv", "text/csv")
    else:
        st.warning("⚠️ Run `python train.py` first.")

# ══════════════════════════════════════════════════════════════
# PAGE: Analytics & EDA
# ══════════════════════════════════════════════════════════════
elif page == "📊 Analytics & EDA":
    st.markdown("## 📊 Exploratory Data Analysis")
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x="plan_tier", y="churn_probability", color="plan_tier",
                         title="Churn Probability by Plan Tier",
                         color_discrete_sequence=px.colors.qualitative.Vivid)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#E8E8F0", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.violin(df, x="churn_segment", y="tenure_months",
                             color="churn_segment", color_discrete_map=SEG_COLOR,
                             title="Tenure Distribution by Risk Segment", box=True)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#E8E8F0", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        plan_seg = df.groupby(["plan_tier","churn_segment"]).size().reset_index(name="count")
        fig3 = px.sunburst(plan_seg, path=["plan_tier","churn_segment"], values="count",
                           title="Plan → Risk Segment Breakdown",
                           color="count", color_continuous_scale="Purples")
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E8E8F0")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("⚠️ Run `python train.py` first.")

# ══════════════════════════════════════════════════════════════
# PAGE: Model Performance
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("## 🤖 Model Performance & Explainability")
    if metrics:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("ROC-AUC", metrics.get("roc_auc","—"))
        c2.metric("PR-AUC", metrics.get("pr_auc","—"))
        c3.metric("Lift @ 10%", f"{metrics.get('lift_at_10pct','—')}x")
        c4.metric("Churn Rate", f"{metrics.get('churn_rate',0)*100:.1f}%")

        model_df = pd.DataFrame({
            "Model": ["Logistic Regression","Random Forest","XGBoost (Calibrated)"],
            "PR-AUC": [metrics.get("logit_pr_auc",0),
                       metrics.get("rf_pr_auc",0),
                       metrics.get("pr_auc",0)]
        })
        fig = px.bar(model_df, x="Model", y="PR-AUC", color="PR-AUC",
                     color_continuous_scale="Purples", title="Model Comparison — PR-AUC",
                     text="PR-AUC")
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#E8E8F0", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    img_path = "images/evaluation_dashboard.png"
    if os.path.exists(img_path):
        st.image(img_path, caption="Full Evaluation Dashboard (from training)", use_column_width=True)
    else:
        st.info("Run `python train.py` to generate evaluation plots.")

    shap_path = "outputs/shap_importance.json"
    if os.path.exists(shap_path):
        with open(shap_path) as f:
            shap_d = json.load(f)
        shap_df = pd.DataFrame(list(shap_d.items()), columns=["Feature","Importance"])
        fig2 = px.bar(shap_df.sort_values("Importance"), x="Importance", y="Feature",
                      orientation="h", title="Top SHAP Feature Importances",
                      color="Importance", color_continuous_scale="Purples")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#E8E8F0")
        st.plotly_chart(fig2, use_container_width=True)
