import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load pipeline and data
model = joblib.load('Models/churn_pipeline.pkl')
df = pd.read_csv('Data/telco_churn.csv')
df.columns = [c.strip().replace(' ', '_').replace('-', '_').lower() for c in df.columns]

# Total charges cleaning (blank spaces ko handle karna)
if 'total_charges' in df.columns:
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    df['total_charges'] = df['total_charges'].fillna(df['total_charges'].median())

# Target extract
if 'churn_value' in df.columns:
    y = pd.to_numeric(df['churn_value'], errors='coerce')
elif 'churn_label' in df.columns:
    y = df['churn_label'].map({'Yes': 1, 'No': 0})
else:
    y = df['churn'].map({'Yes': 1, 'No': 0})

y = y.astype(int)

# Columns drop (same as training)
drop_cols = [
    'customerid', 'count', 'country', 'state', 'city', 'zip_code', 
    'lat_long', 'latitude', 'longitude', 'churn_label', 'churn_value', 
    'churn_score', 'churn_reason', 'target', 'churn'
]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Same test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Probabilities nikalna
probs = model.predict_proba(X_test)[:, 1]

# 40% threshold: Business risk tier (Medium + High Risk)
preds_business = (probs >= 0.40).astype(int)

print("=== OVERALL TEST EVALUATION (1409 Test Customers) ===")
cm = confusion_matrix(y_test, preds_business)
print("\nConfusion Matrix (At 40% Risk Threshold):")
print(f"Actually Stayed & Correctly Identified: {cm[0][0]}")
print(f"Stayed but Flagged at Risk:             {cm[0][1]}")
print(f"Actually Churned but Missed:            {cm[1][0]}")
print(f"Actually Churned & Successfully Caught: {cm[1][1]}")

print("\nDetailed Performance Report:")
print(classification_report(y_test, preds_business, target_names=['Stayed', 'Churned']))