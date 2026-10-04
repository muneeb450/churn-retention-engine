# 🎯 Customer Churn & Retention Engine (SaaS / Telco)

An end-to-end Machine Learning and Business Intelligence solution designed to predict customer attrition risk, analyze core behavioral drivers, and trigger automated, tier-based retention playbooks.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io) 
*(Replace the link above with your live deployed Streamlit URL)*

---

## 📌 Executive Summary & Business Impact
Customer acquisition typically costs **5x more** than customer retention. In recurring subscription businesses (SaaS, Telecom), unaddressed churn directly erodes Annual Recurring Revenue (ARR). 

This project bridges the gap between predictive modeling and commercial execution:
- **Identifies At-Risk Accounts:** Flags high-probability churn customers before contract renewal cycles.
- **Translates Risk into Strategy:** Assigns automated retention playbooks (commercial incentives, technical audits, upsell paths) tailored to customer risk tiers.
- **Scales Decision Making:** Supports both single-account diagnostics and bulk CSV uploads for operations teams.

---

## 🏗️ System Architecture & Workflow

```text
[Raw Telecom Data] 
       │
       ▼
[PostgreSQL Database] ──► Exploratory Analysis & Cohort Segmentation (SQL)
       │
       ▼
[Feature Pipeline]    ──► ColumnTransformer (StandardScaler + OneHotEncoder)
       │
       ▼
[XGBoost Classifier]  ──► Calibrated Probability Scoring & Evaluation
       │
       ▼
[Interactive Web App] ──► Streamlit UI (Single Diagnostics & Batch Scoring Engine)