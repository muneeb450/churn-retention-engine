import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

# 1. Load Data
csv_path = 'Data/telco_churn.csv'
df = pd.read_csv(csv_path)

# Columns clean karein (lowercase + underscore) taake uniform rahein
df.columns = [c.strip().replace(' ', '_').replace('-', '_').lower() for c in df.columns]

# 2. Target Variable & Data Cleaning
# Churn column ko identify karein (churn_value ya churn)
if 'churn_value' in df.columns:
    df['target'] = pd.to_numeric(df['churn_value'], errors='coerce')
elif 'churn_label' in df.columns:
    df['target'] = df['churn_label'].map({'Yes': 1, 'No': 0})
else:
    df['target'] = df['churn'].map({'Yes': 1, 'No': 0})

# Total charges ko numeric cast karein aur missing median se fill karein
if 'total_charges' in df.columns:
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    df['total_charges'] = df['total_charges'].fillna(df['total_charges'].median())

# Unnecessary identifiers aur high-cardinality/leaky columns drop karein
drop_cols = [
    'customerid', 'count', 'country', 'state', 'city', 'zip_code', 
    'lat_long', 'latitude', 'longitude', 'churn_label', 'churn_value', 
    'churn_score', 'churn_reason', 'target', 'churn'
]
cols_to_drop = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=cols_to_drop)
y = df['target'].astype(int)

# 3. Features Categorization
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical.")

# 4. Pipeline Setup
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# Class imbalance weight handle karna
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric='logloss'
    ))
])

# 5. Train & Evaluate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\n--- MODEL PERFORMANCE ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# 6. Save Model Artifact
os.makedirs('Models', exist_ok=True)
joblib.dump(pipeline, 'Models/churn_pipeline.pkl')
print("\nPipeline successfully saved to 'Models/churn_pipeline.pkl'!")