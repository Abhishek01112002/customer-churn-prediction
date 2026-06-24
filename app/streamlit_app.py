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
    page_title="Telco Customer Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.25rem;
        border-radius: 0.75rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6B7280;
        font-weight: 500;
    }
    .strategy-box {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE & INITIALIZATION -----------------

@st.cache_resource
def get_local_model():
    return load_pipeline()

local_pipeline = get_local_model()

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/clouds/200/database.png", width=100)
st.sidebar.title("Configuration Hub")

mode = st.sidebar.selectbox(
    "Execution Mode", 
    ["Standalone (Direct Load)", "REST API Client (FastAPI)"],
    help="Standalone runs inference directly in-process. REST API Client routes requests to a FastAPI server."
)

api_url = "http://localhost:8000"
if mode == "REST API Client (FastAPI)":
    api_url = st.sidebar.text_input("FastAPI Endpoint URL", value="http://localhost:8000")
    
    # Ping API to verify status
    try:
        r = requests.get(f"{api_url}/", timeout=2)
        if r.status_code == 200:
            st.sidebar.success("✅ Connected to FastAPI Backend")
        else:
            st.sidebar.warning("⚠️ API online, but returned error")
    except requests.exceptions.RequestException:
        st.sidebar.error("❌ Cannot connect to FastAPI server. Please check that it is running, or switch to Standalone mode.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🧠 About the System
- **Model Type**: Logistic Regression (SMOTE balanced)
- **Target**: Churn (Yes/No)
- **Optimization Metric**: F1 Score (balances Precision and Recall to minimize customer loss cost).
""")

# Main Page Header
st.markdown('<div class="main-title">📊 Telco Customer Churn Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A production-grade, end-to-end ML platform built with modular software engineering. Separates backend API services from dashboard components.</div>', unsafe_allow_html=True)

# Tabs configuration
tab1, tab2, tab3 = st.tabs(["👤 Single Customer Inference", "📂 Bulk CSV Batch Analytics", "⚙️ Model Performance & Diagnostics"])

# ----------------- TAB 1: SINGLE CUSTOMER INFERENCE -----------------
with tab1:
    st.markdown("### Profile Constructor")
    st.write("Construct a customer profile in the panels below to calculate real-time churn risk.")
    
    # Input panels
    col_demo, col_account, col_services = st.columns(3)
    
    with col_demo:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen (Age >= 65)", [0, 1])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        
    with col_account:
        st.subheader("Billing & Account")
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
        st.subheader("Services Subscribed")
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
    
    # Prediction logic
    customer_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    col_act, col_res = st.columns([1, 2])
    
    with col_act:
        st.write("")
        st.write("")
        predict_btn = st.button("Run Churn Risk Analysis", type="primary", use_container_width=True)
        
    with col_res:
        if predict_btn:
            churn_probability = 0.0
            risk_level = "Low"
            prediction = 0
            
            # Executing based on Mode selection
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
            
            # Display prediction card
            prob_pct = churn_probability * 100
            
            st.markdown("### Inference Summary")
            
            if risk_level == "High":
                st.error(f"🚨 **High Risk Profile** - {prob_pct:.1f}% Churn Probability")
            elif risk_level == "Medium":
                st.warning(f"⚠️ **Medium Risk Profile** - {prob_pct:.1f}% Churn Probability")
            else:
                st.success(f"✅ **Low Risk Profile** - {prob_pct:.1f}% Churn Probability")
                
            st.progress(churn_probability)
            
            # Display business strategies
            st.markdown('<div class="strategy-box">', unsafe_allow_html=True)
            st.markdown("#### 💡 Tactical Retention Strategy Recommendation")
            if risk_level == "High":
                st.markdown("""
                - **Immediate Action**: Active outreach. Customer has high churn propensity.
                - **Offer**: 20% discount on a 1-year contract extension OR offer 2 months free support services.
                - **Primary Churn Drivers**: Month-to-month terms, High Monthly Charges, or Lack of Online Tech Support.
                """)
            elif risk_level == "Medium":
                if tenure < 12:
                    st.markdown("""
                    - **Onboarding Risk**: Low tenure customer experiencing friction.
                    - **Action**: Automate a CS touchpoint email detailing self-help portals and product usage tips.
                    - **Offer**: A free month of Online Security/Backup services.
                    """)
                else:
                    st.markdown("""
                    - **Action**: Check account health metrics. Monitor usage drop-off.
                    - **Offer**: Offer device protection or fiber optic upgrade package discounts.
                    """)
            else:
                st.markdown("""
                - **Account Status**: Strong loyalty, low risk.
                - **Action**: Target for premium service cross-selling (e.g. streaming packages, multi-line additions).
                - **Strategy**: Send an annual loyalty reward thank you note.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

# ----------------- TAB 2: BATCH CSV INFERENCE -----------------
with tab2:
    st.markdown("### Bulk CSV Batch Inference")
    st.write("Upload a batch CSV list of customer accounts to score customer churn at scale.")
    
    # Download sample block
    sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sample_customers.csv')
    if os.path.exists(sample_path):
        with open(sample_path, "r") as f:
            sample_csv_data = f.read()
        st.download_button(
            label="📥 Download Template Sample CSV",
            data=sample_csv_data,
            file_name="sample_customers.csv",
            mime="text/csv"
        )
        
    uploaded_file = st.file_uploader("Drag and Drop CSV File", type="csv")
    
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.markdown("#### Input Data Preview")
        st.dataframe(df_uploaded.head(5))
        
        batch_btn = st.button("Score Uploaded Customers", type="primary")
        
        if batch_btn:
            scored_df = None
            
            # API Mode
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
                    st.warning(f"Could not reach FastAPI API: {e}. Falling back to standalone python processing.")
                    if local_pipeline is not None:
                        scored_df = predict_batch_customers(df_uploaded, local_pipeline)
                    else:
                        st.stop()
            # Standalone Mode
            else:
                if local_pipeline is not None:
                    with st.spinner("Scoring customer data locally..."):
                        scored_df = predict_batch_customers(df_uploaded, local_pipeline)
                else:
                    st.error("Trained model pipeline not found. Run training script first.")
                    st.stop()
            
            if scored_df is not None:
                st.success("🎉 Batch Customer Scoring Completed Successfully!")
                
                # Show key metrics side by side
                tot_cust = len(scored_df)
                high_risk = len(scored_df[scored_df['risk_level'] == 'High'])
                med_risk = len(scored_df[scored_df['risk_level'] == 'Medium'])
                low_risk = len(scored_df[scored_df['risk_level'] == 'Low'])
                churn_rate = (scored_df['prediction'].sum() / tot_cust) * 100
                
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{tot_cust}</div><div class="metric-label">Scored Customers</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{churn_rate:.1f}%</div><div class="metric-label">Overall Churn Rate</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="metric-card" style="border-left-color: #EF4444;"><div class="metric-value">{high_risk}</div><div class="metric-label">High Churn Risk Customers</div></div>', unsafe_allow_html=True)
                with m4:
                    st.markdown(f'<div class="metric-card" style="border-left-color: #10B981;"><div class="metric-value">{low_risk}</div><div class="metric-label">Low Churn Risk Customers</div></div>', unsafe_allow_html=True)
                
                # Plotly Charts
                g1, g2 = st.columns(2)
                
                with g1:
                    risk_counts = scored_df['risk_level'].value_counts().reset_index()
                    risk_counts.columns = ['Risk Level', 'Count']
                    fig_pie = px.pie(
                        risk_counts, 
                        values='Count', 
                        names='Risk Level', 
                        title='Customer Churn Risk Breakdown',
                        color='Risk Level',
                        color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with g2:
                    fig_scatter = px.scatter(
                        scored_df, 
                        x='tenure', 
                        y='churn_probability', 
                        color='risk_level',
                        hover_data=['MonthlyCharges', 'Contract'],
                        title='Tenure vs Churn Propensity Scatter Plot',
                        labels={'tenure': 'Tenure (Months)', 'churn_probability': 'Churn Propensity (%)'},
                        color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Show results dataframe
                st.markdown("#### Scored Dataset Results")
                st.dataframe(scored_df.style.background_gradient(subset=['churn_probability'], cmap='Reds'))
                
                # Download link
                csv_output = scored_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Full Predictions CSV Output",
                    data=csv_output,
                    file_name="scored_customers_output.csv",
                    mime="text/csv"
                )

# ----------------- TAB 3: DIAGNOSTICS & METRICS -----------------
with tab3:
    st.markdown("### Model Diagnostics & Calibration")
    st.write("Inspect offline validation performance, loss functions, metrics, and models serialization history.")
    
    # Read metrics.json
    metrics_path = os.path.join(config.MODEL_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_payload = json.load(f)
            
        best_model_name = metrics_payload['best_model']
        best_metrics = metrics_payload['best_model_metrics']
        comparison_list = metrics_payload['comparison']
        
        # Display best model
        st.info(f"🏆 **Selected Production Model**: **{best_model_name}** (optimized for F1-Score to offset cost of false negatives)")
        
        # Metrics cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric(label="Accuracy Score", value=f"{best_metrics['Accuracy']*100:.2f}%")
        with c2:
            st.metric(label="F1 Score", value=f"{best_metrics['F1']*100:.2f}%")
        with c3:
            st.metric(label="Recall (Churn Caught)", value=f"{best_metrics['Recall']*100:.2f}%")
        with c4:
            st.metric(label="Precision (Accuracy on Flag)", value=f"{best_metrics['Precision']*100:.2f}%")
        with c5:
            st.metric(label="ROC-AUC Score", value=f"{best_metrics['ROC-AUC']*100:.2f}%")
            
        # Model Comparison Table
        st.markdown("#### Cross-Validated Model Evaluation Summary")
        comp_df = pd.DataFrame(comparison_list).set_index('Model')
        st.dataframe(comp_df.style.highlight_max(color='#D1FAE5', axis=0))
        
    else:
        st.warning("Diagnostics metrics.json file not found. Run training process to compile model diagnostics.")
        
    # Image Plot displays
    st.markdown("#### Offline Evaluation Visualizations")
    img_col1, img_col2, img_col3 = st.columns(3)
    
    cm_img_path = os.path.join(config.MODEL_DIR, 'confusion_matrix.png')
    roc_img_path = os.path.join(config.MODEL_DIR, 'roc_curve.png')
    fi_img_path = os.path.join(config.MODEL_DIR, 'feature_importance.png')
    
    with img_col1:
        st.subheader("Confusion Matrix")
        if os.path.exists(cm_img_path):
            st.image(Image.open(cm_img_path), use_container_width=True)
        else:
            st.write("Confusion Matrix plot not available.")
            
    with img_col2:
        st.subheader("ROC Curve")
        if os.path.exists(roc_img_path):
            st.image(Image.open(roc_img_path), use_container_width=True)
        else:
            st.write("ROC Curve plot not available.")
            
    with img_col3:
        st.subheader("Feature Importances")
        if os.path.exists(fi_img_path):
            st.image(Image.open(fi_img_path), use_container_width=True)
        else:
            st.write("Feature Importance plot not available for this model type.")
