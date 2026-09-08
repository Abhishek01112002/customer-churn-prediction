import os
import sys
import json
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Add src to python path for modular imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import load_pipeline, predict_single_customer, predict_batch_customers, get_feature_importance
from src import config

# Page setup
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════
#  APPLE-INSPIRED DESIGN SYSTEM (Conservative, layout-safe CSS)
# ══════════════════════════════════════════════════════════════════════

# Load Inter font via link tag (not @import inside style)
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

st.markdown("""
<style>
    /* ─── Global Typography & Background ─── */
    html, body, .stApp,
    h1, h2, h3, h4, h5, h6,
    p, li, span, label, div, input, textarea, select, button {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                     'SF Pro Display', 'Segoe UI', 'Helvetica Neue', sans-serif !important;
    }
    .stApp {
        background-color: #F5F5F7 !important;
    }
    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1C1C1E 0%, #2C2C2E 100%) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] li {
        color: rgba(255,255,255,0.85) !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(255,255,255,0.07) !important;
        border-color: rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div span {
        color: rgba(255,255,255,0.9) !important;
    }
    section[data-testid="stSidebar"] .stTextInput > div > div > input {
        background: rgba(255,255,255,0.07) !important;
        border-color: rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: white !important;
    }

    /* ─── Primary Buttons ─── */
    button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(180deg, #007AFF 0%, #0063D1 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 4px 14px rgba(0,122,255,0.2);
        transition: all 0.2s ease;
    }
    button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        box-shadow: 0 6px 20px rgba(0,122,255,0.3) !important;
        opacity: 0.95;
    }

    /* ─── Download Buttons ─── */
    .stDownloadButton > button {
        background: rgba(0,122,255,0.06) !important;
        color: #007AFF !important;
        border: 1.5px solid rgba(0,122,255,0.18) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(0,122,255,0.1) !important;
    }

    /* ─── Form Inputs (light touch) ─── */
    .stSelectbox > div > div {
        border-radius: 10px !important;
    }
    .stNumberInput > div > div > input {
        border-radius: 10px !important;
    }
    .stSlider > div > div > div > div {
        background-color: #007AFF !important;
    }

    /* ─── Progress Bar ─── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #007AFF, #5AC8FA) !important;
        border-radius: 8px !important;
    }
    .stProgress > div > div > div {
        border-radius: 8px !important;
    }

    /* ─── Alerts ─── */
    .stAlert > div {
        border-radius: 12px !important;
    }

    /* ─── Data Tables ─── */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* ─── Images ─── */
    .stImage img {
        border-radius: 12px;
    }

    /* ─── st.metric glass effect ─── */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid rgba(0,0,0,0.04);
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #86868B !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* ─── Hide default branding ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ══════════ CUSTOM COMPONENT CLASSES ══════════ */

    /* Hero header */
    .apple-hero {
        text-align: center;
        padding: 1.5rem 1rem 0.75rem;
    }
    .apple-hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        line-height: 1.15;
        color: #1D1D1F;
        margin-bottom: 0.5rem;
    }
    .apple-hero-sub {
        font-size: 1.05rem;
        color: #86868B;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Glass metric cards */
    .glass-metric {
        background: white;
        border: 1px solid rgba(0,0,0,0.05);
        border-radius: 18px;
        padding: 1.25rem 1rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .glass-metric:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    }
    .glass-metric .gm-icon { font-size: 1.5rem; margin-bottom: 0.3rem; }
    .glass-metric .gm-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1D1D1F;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }
    .glass-metric .gm-label {
        font-size: 0.68rem;
        color: #86868B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-top: 0.25rem;
    }

    /* Retention strategy card */
    .strategy-card {
        background: linear-gradient(135deg, rgba(0,122,255,0.04) 0%, rgba(90,200,250,0.02) 100%);
        border: 1px solid rgba(0,122,255,0.1);
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-top: 1rem;
    }
    .strategy-card h4 {
        color: #007AFF !important;
        font-weight: 600 !important;
    }

    /* Section pill label */
    .section-pill {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #007AFF;
        background: rgba(0,122,255,0.08);
        padding: 4px 14px;
        border-radius: 20px;
        margin-bottom: 0.25rem;
    }

    /* Sidebar brand block */
    .sidebar-brand {
        text-align: center;
        padding: 1rem 0 0.25rem;
    }
    .sidebar-brand .sb-icon { font-size: 2rem; }
    .sidebar-brand .sb-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: white !important;
        letter-spacing: -0.02em;
        margin-top: 0.3rem;
    }
    .sidebar-brand .sb-ver {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.3) !important;
        margin-top: 0.1rem;
    }

    /* Sidebar about card */
    .sidebar-about {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem 1.1rem;
    }
    .sidebar-about .sa-title {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: rgba(255,255,255,0.35) !important;
        margin-bottom: 0.6rem;
    }
    .sidebar-about .sa-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .sidebar-about .sa-row:last-child { border-bottom: none; }
    .sidebar-about .sa-key { color: rgba(255,255,255,0.45) !important; }
    .sidebar-about .sa-val { font-weight: 600; color: rgba(255,255,255,0.85) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  SESSION STATE & MODEL INITIALIZATION
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_local_model():
    return load_pipeline()

local_pipeline = get_local_model()

# Plotly Apple-Style Layout
PLOTLY_APPLE = dict(
    font=dict(
        family="Inter, -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif",
        color="#1D1D1F",
        size=13,
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=52, l=48, r=20, b=48),
    title_font=dict(size=16, color="#1D1D1F"),
    legend=dict(font=dict(size=12, color="#6E6E73"), bgcolor="rgba(0,0,0,0)", borderwidth=0),
    hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter, sans-serif", bordercolor="#E5E5EA"),
)
APPLE_RISK_COLORS = {'Low': '#30D158', 'Medium': '#FF9F0A', 'High': '#FF453A'}

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════

st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="sb-icon">⚡</div>
    <div class="sb-name">Churn Predictor</div>
    <div class="sb-ver">v2.0 · Production</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

mode = st.sidebar.selectbox(
    "Execution Mode",
    ["Standalone (Direct Load)", "REST API Client (FastAPI)"],
    help="Standalone runs inference directly in-process. REST API Client routes requests to a FastAPI server."
)

api_url = "http://localhost:8000"
if mode == "REST API Client (FastAPI)":
    api_url = st.sidebar.text_input("FastAPI Endpoint URL", value="http://localhost:8000")
    try:
        r = requests.get(f"{api_url}/", timeout=2)
        if r.status_code == 200:
            st.sidebar.success("✅ Connected to FastAPI Backend")
        else:
            st.sidebar.warning("⚠️ API online, but returned error")
    except requests.exceptions.RequestException:
        st.sidebar.error("❌ Cannot connect to FastAPI server. Switch to Standalone mode.")

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div class="sidebar-about">
    <div class="sa-title">System Details</div>
    <div class="sa-row"><span class="sa-key">Model</span><span class="sa-val">Logistic Regression</span></div>
    <div class="sa-row"><span class="sa-key">Balancing</span><span class="sa-val">SMOTE</span></div>
    <div class="sa-row"><span class="sa-key">Target</span><span class="sa-val">Churn (Yes / No)</span></div>
    <div class="sa-row"><span class="sa-key">Metric</span><span class="sa-val">F1 Score</span></div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  MAIN HEADER
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="apple-hero">
    <div class="apple-hero-title">⚡ Customer Churn Prediction</div>
    <div class="apple-hero-sub">
        Production-grade ML platform for real-time customer risk scoring,
        batch analytics, and intelligent retention strategies.
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["👤 Single Prediction", "📂 Batch Analytics", "⚙️ Model Diagnostics"])

# ─────────────────── TAB 1: SINGLE CUSTOMER ───────────────────
with tab1:
    st.markdown('<span class="section-pill">Customer Profile</span>', unsafe_allow_html=True)
    st.markdown("### Build a Customer Profile")
    st.caption("Configure the parameters below to calculate real-time churn risk.")

    col_demo, col_account, col_services = st.columns(3)

    with col_demo:
        st.markdown("##### Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen (Age ≥ 65)", [0, 1])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    with col_account:
        st.markdown("##### Billing & Account")
        tenure = st.slider("Tenure (Months)", 0, 72, 24)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total_charges = st.number_input("Total Charges ($)", 18.0, 8500.0, 1500.0)

    with col_services:
        st.markdown("##### Services Subscribed")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"] if phone_service == "Yes" else ["No phone service"])
        internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])

        if internet_service != "No":
            online_security = st.selectbox("Online Security", ["No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
        else:
            online_security = online_backup = device_protection = tech_support = streaming_tv = streaming_movies = "No internet service"

    st.markdown("---")

    customer_data = {
        'gender': gender, 'SeniorCitizen': senior_citizen,
        'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone_service,
        'MultipleLines': multiple_lines, 'InternetService': internet_service,
        'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
        'DeviceProtection': device_protection, 'TechSupport': tech_support,
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }

    col_act, col_res = st.columns([1, 2])

    with col_act:
        st.write("")
        st.write("")
        predict_btn = st.button("⚡ Analyze Churn Risk", type="primary", use_container_width=True)

    with col_res:
        if predict_btn:
            churn_probability = 0.0
            risk_level = "Low"
            prediction = 0

            if mode == "REST API Client (FastAPI)":
                try:
                    response = requests.post(f"{api_url}/predict", json=customer_data, timeout=5)
                    if response.status_code == 200:
                        res_json = response.json()
                        churn_probability = res_json['churn_probability']
                        risk_level = res_json['risk_level']
                        prediction = res_json['prediction']
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                        st.stop()
                except Exception as e:
                    st.error(f"Failed to query FastAPI backend: {e}. Falling back to stand-alone engine...")
                    if local_pipeline is not None:
                        res_local = predict_single_customer(customer_data, local_pipeline)
                        churn_probability = res_local['churn_probability']
                        risk_level = res_local['risk_level']
                        prediction = res_local['prediction']
                    else:
                        st.stop()
            else:
                if local_pipeline is not None:
                    res_local = predict_single_customer(customer_data, local_pipeline)
                    churn_probability = res_local['churn_probability']
                    risk_level = res_local['risk_level']
                    prediction = res_local['prediction']
                else:
                    st.error("Local pipeline file not found. Run training script first.")
                    st.stop()

            prob_pct = churn_probability * 100

            st.markdown('<span class="section-pill">Inference Result</span>', unsafe_allow_html=True)
            st.markdown("### Risk Assessment")

            if risk_level == "High":
                st.error(f"🚨 **High Risk** — {prob_pct:.1f}% churn probability detected")
            elif risk_level == "Medium":
                st.warning(f"⚠️ **Medium Risk** — {prob_pct:.1f}% churn probability detected")
            else:
                st.success(f"✅ **Low Risk** — {prob_pct:.1f}% churn probability")

            st.progress(churn_probability)

            st.markdown('<div class="strategy-card">', unsafe_allow_html=True)
            st.markdown("#### 💡 Retention Strategy")
            if risk_level == "High":
                st.markdown("""
                - **Immediate Action**: Active outreach — customer has high churn propensity.
                - **Offer**: 20% discount on 1-year extension OR 2 months free support.
                - **Primary Drivers**: Month-to-month terms, high charges, or lack of tech support.
                """)
            elif risk_level == "Medium":
                if tenure < 12:
                    st.markdown("""
                    - **Onboarding Risk**: Low tenure customer experiencing friction.
                    - **Action**: Automate CS touchpoint email with self-help portals.
                    - **Offer**: Free month of Online Security/Backup services.
                    """)
                else:
                    st.markdown("""
                    - **Action**: Check account health metrics. Monitor usage drop-off.
                    - **Offer**: Device protection or fiber optic upgrade discounts.
                    """)
            else:
                st.markdown("""
                - **Account Status**: Strong loyalty, low risk.
                - **Action**: Target for premium cross-selling (streaming, multi-line).
                - **Strategy**: Send annual loyalty reward note.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────── TAB 2: BATCH CSV ───────────────────
with tab2:
    st.markdown('<span class="section-pill">Batch Processing</span>', unsafe_allow_html=True)
    st.markdown("### Bulk Customer Scoring")
    st.caption("Upload a CSV of customer accounts to score churn risk at scale.")

    sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sample_customers.csv')
    if os.path.exists(sample_path):
        with open(sample_path, "r") as f:
            sample_csv_data = f.read()
        st.download_button(
            label="📥 Download Template CSV",
            data=sample_csv_data,
            file_name="sample_customers.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader("Drag and drop a CSV file", type="csv")

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.markdown("#### Input Preview")
        st.dataframe(df_uploaded.head(5))

        batch_btn = st.button("⚡ Score All Customers", type="primary")

        if batch_btn:
            scored_df = None

            if mode == "REST API Client (FastAPI)":
                try:
                    with st.spinner("Calling FastAPI batch engine..."):
                        files = {'file': uploaded_file.getvalue()}
                        response = requests.post(f"{api_url}/predict/batch/csv", files=files, timeout=10)
                        if response.status_code == 200:
                            preds = response.json()['predictions']
                            scored_df = df_uploaded.copy()
                            scored_df['churn_probability'] = [p['churn_probability'] for p in preds]
                            scored_df['prediction'] = [p['prediction'] for p in preds]
                            scored_df['risk_level'] = [p['risk_level'] for p in preds]
                        else:
                            st.error(f"FastAPI Batch Error: {response.text}")
                            st.stop()
                except Exception as e:
                    st.warning(f"Could not reach FastAPI API: {e}. Falling back to standalone processing.")
                    if local_pipeline is not None:
                        scored_df = predict_batch_customers(df_uploaded, local_pipeline)
                    else:
                        st.stop()
            else:
                if local_pipeline is not None:
                    with st.spinner("Scoring customer data locally..."):
                        scored_df = predict_batch_customers(df_uploaded, local_pipeline)
                else:
                    st.error("Trained model pipeline not found. Run training script first.")
                    st.stop()

            if scored_df is not None:
                st.success("Batch scoring completed successfully.")

                tot_cust = len(scored_df)
                high_risk = len(scored_df[scored_df['risk_level'] == 'High'])
                med_risk = len(scored_df[scored_df['risk_level'] == 'Medium'])
                low_risk = len(scored_df[scored_df['risk_level'] == 'Low'])
                churn_rate = (scored_df['prediction'].sum() / tot_cust) * 100

                # Glass Metric Cards
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f'<div class="glass-metric"><div class="gm-icon">👥</div><div class="gm-value">{tot_cust}</div><div class="gm-label">Customers Scored</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="glass-metric"><div class="gm-icon">📊</div><div class="gm-value">{churn_rate:.1f}%</div><div class="gm-label">Overall Churn Rate</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="glass-metric"><div class="gm-icon">🔴</div><div class="gm-value">{high_risk}</div><div class="gm-label">High Risk</div></div>', unsafe_allow_html=True)
                with m4:
                    st.markdown(f'<div class="glass-metric"><div class="gm-icon">🟢</div><div class="gm-value">{low_risk}</div><div class="gm-label">Low Risk</div></div>', unsafe_allow_html=True)

                st.write("")

                # Plotly Charts
                g1, g2 = st.columns(2)

                with g1:
                    risk_counts = scored_df['risk_level'].value_counts().reset_index()
                    risk_counts.columns = ['Risk Level', 'Count']
                    fig_pie = px.pie(
                        risk_counts, values='Count', names='Risk Level',
                        title='Risk Distribution',
                        color='Risk Level',
                        color_discrete_map=APPLE_RISK_COLORS,
                        hole=0.45,
                    )
                    fig_pie.update_layout(**PLOTLY_APPLE)
                    fig_pie.update_traces(
                        textfont_size=13, textinfo='percent+label',
                        marker=dict(line=dict(color='#FFFFFF', width=2)),
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with g2:
                    fig_scatter = px.scatter(
                        scored_df, x='tenure', y='churn_probability',
                        color='risk_level',
                        hover_data=['MonthlyCharges', 'Contract'],
                        title='Tenure vs Churn Propensity',
                        labels={'tenure': 'Tenure (months)', 'churn_probability': 'Churn Probability'},
                        color_discrete_map=APPLE_RISK_COLORS,
                    )
                    fig_scatter.update_layout(**PLOTLY_APPLE)
                    fig_scatter.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1.5, color='white')))
                    fig_scatter.update_xaxes(gridcolor='rgba(0,0,0,0.04)', zerolinecolor='rgba(0,0,0,0.06)')
                    fig_scatter.update_yaxes(gridcolor='rgba(0,0,0,0.04)', zerolinecolor='rgba(0,0,0,0.06)')
                    st.plotly_chart(fig_scatter, use_container_width=True)

                st.markdown("#### Scored Results")
                st.dataframe(scored_df.style.background_gradient(subset=['churn_probability'], cmap='Reds'))

                csv_output = scored_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Predictions CSV",
                    data=csv_output,
                    file_name="scored_customers_output.csv",
                    mime="text/csv"
                )

# ─────────────────── TAB 3: MODEL DIAGNOSTICS ───────────────────
with tab3:
    st.markdown('<span class="section-pill">Diagnostics</span>', unsafe_allow_html=True)
    st.markdown("### Model Performance & Calibration")
    st.caption("Inspect offline validation metrics, confusion matrices, and feature importances.")

    metrics_path = os.path.join(config.MODEL_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_payload = json.load(f)

        best_model_name = metrics_payload['best_model']
        best_metrics = metrics_payload['best_model_metrics']
        comparison_list = metrics_payload['comparison']

        st.info(f"🏆 **Production Model**: **{best_model_name}** — optimized for F1-Score to offset cost of false negatives.")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric(label="Accuracy", value=f"{best_metrics['Accuracy']*100:.2f}%")
        with c2:
            st.metric(label="F1 Score", value=f"{best_metrics['F1']*100:.2f}%")
        with c3:
            st.metric(label="Recall", value=f"{best_metrics['Recall']*100:.2f}%")
        with c4:
            st.metric(label="Precision", value=f"{best_metrics['Precision']*100:.2f}%")
        with c5:
            st.metric(label="ROC-AUC", value=f"{best_metrics['ROC-AUC']*100:.2f}%")

        st.write("")
        st.markdown("#### Cross-Validated Evaluation")
        comp_df = pd.DataFrame(comparison_list).set_index('Model')
        st.dataframe(comp_df.style.highlight_max(color='#D1FAE5', axis=0))
    else:
        st.warning("Diagnostics metrics.json not found. Run training process first.")

    st.write("")
    st.markdown("#### Evaluation Visualizations")
    img_col1, img_col2, img_col3 = st.columns(3)

    cm_img_path = os.path.join(config.MODEL_DIR, 'confusion_matrix.png')
    roc_img_path = os.path.join(config.MODEL_DIR, 'roc_curve.png')
    fi_img_path = os.path.join(config.MODEL_DIR, 'feature_importance.png')

    with img_col1:
        st.markdown("##### Confusion Matrix")
        if os.path.exists(cm_img_path):
            st.image(Image.open(cm_img_path), use_container_width=True)
        else:
            st.caption("Plot not available.")

    with img_col2:
        st.markdown("##### ROC Curve")
        if os.path.exists(roc_img_path):
            st.image(Image.open(roc_img_path), use_container_width=True)
        else:
            st.caption("Plot not available.")

    with img_col3:
        st.markdown("##### Feature Importances")
        if os.path.exists(fi_img_path):
            st.image(Image.open(fi_img_path), use_container_width=True)
        else:
            st.caption("Plot not available for this model type.")
