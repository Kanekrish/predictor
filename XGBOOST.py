import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

DATASET_PATH = "seer_breast_cancer.csv"
TARGET_COLUMN = "6th Stage"

df = pd.read_csv(r"C:\dataset.csv")
df.columns = df.columns.str.strip()

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found!")

df = df.dropna(subset=[TARGET_COLUMN]).copy()

# Remove a completely empty unnamed column if present.
empty_unnamed = [
    c for c in df.columns
    if c.lower().startswith("unnamed") and df[c].isna().all()
]
if empty_unnamed:
    df = df.drop(columns=empty_unnamed)

X = df.drop(columns=[TARGET_COLUMN]).copy()
y = df[TARGET_COLUMN].astype(str)

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

feature_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    feature_encoders[col] = le

feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X.values,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(
    y_test, y_pred,
    target_names=target_encoder.classes_,
    zero_division=0
))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save every artifact needed by the deployed application.
joblib.dump(model, "xgboost_model.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
joblib.dump(feature_encoders, "feature_encoders.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

print("Saved:")
print("xgboost_model.pkl")
print("target_encoder.pkl")
print("feature_encoders.pkl")
print("feature_columns.pkl")
