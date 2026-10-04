import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Customer Churn & Retention Engine",
    page_icon="📊",
    layout="wide"
)

# Load trained pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load('Models/churn_pipeline.pkl')

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'Models/churn_pipeline.pkl' exists.")
    st.stop()

# Header & Overview
st.title("🎯 Customer Churn & Retention Engine")
st.markdown("Predict customer attrition risk and trigger proactive retention workflows.")

# Tabs for Single Predictor & Batch Prediction
tab1, tab2 = st.tabs(["👤 Single Customer Risk Assessor", "📁 Batch Customer Analysis"])

# ================= TAB 1: Single Prediction =================
with tab1:
    st.subheader("Customer Demographics & Account Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure_months = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col3:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=65.0)

    # Estimate total charges based on tenure & monthly rate
    total_charges = round(tenure_months * monthly_charges, 2)
    cltv = int(monthly_charges * tenure_months * 0.8) # approximate CLTV if needed

    st.write("---")

    if st.button("🚀 Analyze Churn Risk", type="primary"):
        # Match features with model pipeline expected columns
        input_data = pd.DataFrame([{
            'gender': gender,
            'senior_citizen': senior_citizen,
            'partner': partner,
            'dependents': dependents,
            'tenure_months': tenure_months,
            'phone_service': phone_service,
            'multiple_lines': multiple_lines,
            'internet_service': internet_service,
            'online_security': online_security,
            'online_backup': online_backup,
            'device_protection': device_protection,
            'tech_support': tech_support,
            'streaming_tv': streaming_tv,
            'streaming_movies': streaming_movies,
            'contract': contract,
            'paperless_billing': paperless_billing,
            'payment_method': payment_method,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'cltv': cltv
        }])

        prob = pipeline.predict_proba(input_data)[0][1]
        risk_pct = round(prob * 100, 2)

        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label="Predicted Churn Probability", value=f"{risk_pct}%")
            if risk_pct >= 70:
                st.error("🚨 Risk Tier: **HIGH RISK**")
            elif 40 <= risk_pct < 70:
                st.warning("⚠️ Risk Tier: **MEDIUM RISK**")
            else:
                st.success("✅ Risk Tier: **LOW RISK (Healthy)**")

        with res_col2:
            st.subheader("💡 Automated Retention Strategy")
            if risk_pct >= 70:
                st.markdown("""
                - **Urgent Action:** Assign Dedicated Account Manager within 24 hours.
                - **Commercial Offer:** Provide a **15% renewal discount** on committing to an Annual contract.
                - **Quality Audit:** Review last 3 support tickets to address unresolved service bottlenecks.
                """)
            elif 40 <= risk_pct < 70:
                st.markdown("""
                - **Nurture Action:** Enroll customer into a targeted feature-adoption email workflow.
                - **Value Incentive:** Offer complimentary Tech Support / Security add-on for 2 months.
                - **Feedback:** Trigger automated 1-click CSAT survey to understand pain points.
                """)
            else:
                st.markdown("""
                - **Upsell Candidate:** Eligible for high-speed fiber upgrade or premium streaming bundle.
                - **Advocacy:** Invite to join customer referral reward program.
                """)

# ================= TAB 2: Batch Analysis =================
with tab2:
    st.subheader("Upload Customer Dataset for Bulk Assessment")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        batch_clean = batch_df.copy()
        batch_clean.columns = [c.strip().replace(' ', '_').replace('-', '_').lower() for c in batch_clean.columns]
        
        st.write(f"Uploaded **{len(batch_clean)}** customer records.")
        
        if st.button("Run Batch Prediction"):
            with st.spinner("Processing customer risks..."):
                try:
                    probs = pipeline.predict_proba(batch_clean)[:, 1]
                    batch_df['Churn_Probability_%'] = (probs * 100).round(2)
                    batch_df['Risk_Tier'] = batch_df['Churn_Probability_%'].apply(
                        lambda p: 'High Risk' if p >= 70 else ('Medium Risk' if p >= 40 else 'Low Risk')
                    )
                    
                    st.success("Batch Prediction Complete!")
                    st.dataframe(batch_df[['Churn_Probability_%', 'Risk_Tier'] + [c for c in batch_df.columns if c not in ['Churn_Probability_%', 'Risk_Tier']]].head(15))
                    
                    csv_export = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Enriched Predictions (CSV)",
                        data=csv_export,
                        file_name="churn_predictions_output.csv",
                        mime="text/csv"
                    )
                except Exception as err:
                    st.error(f"Error during batch scoring: {err}. Ensure required feature columns are present.")