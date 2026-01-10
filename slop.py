"""
Hajj/Umrah Crowd Management - Anomaly Detection using Isolation Forest
====================================================================
Goal:
Detect abnormal crowd behavior (sudden movements, congestion, stress)
and trigger security response using anomaly detection + decision logic.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# STEP 1: DATA LOADING
# ============================================================

try:
    df = pd.read_csv("hajj_umrah_crowd_management_dataset.csv", encoding="utf-8", on_bad_lines="skip")
except:
    df = pd.read_csv("hajj_umrah_crowd_management_dataset.csv", encoding="latin1", on_bad_lines="skip")

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# STEP 2: PROXY GROUND TRUTH (FOR EVALUATION ONLY)
# NOTE: Isolation Forest is UNSUPERVISED.
# Labels below are NOT used for training, only evaluation.
# ============================================================

df["Is_Abnormal"] = 0
# Convert required columns to numeric safely
cols = [
    "Movement_Speed", "Sound_Level_dB", "Crowd_Density",
    "Distance_Between_People_m", "Queue_Time_minutes", "Stress_Level"
]

for col in cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Combination-based abnormality (NOT single-feature spikes)
abnormal_conditions = (
    ((df["Crowd_Density"] > df["Crowd_Density"].quantile(0.90)) &
     (df["Sound_Level_dB"] > df["Sound_Level_dB"].quantile(0.90)))
    |
    ((df["Movement_Speed"] < df["Movement_Speed"].quantile(0.05)) &
     (df["Distance_Between_People_m"] < df["Distance_Between_People_m"].quantile(0.10)))
    |
    (df["Queue_Time_minutes"] > df["Queue_Time_minutes"].quantile(0.95))
)

df.loc[abnormal_conditions, "Is_Abnormal"] = 1

actual_anomaly_rate = df["Is_Abnormal"].mean()
print(f"Ground truth anomalies: {actual_anomaly_rate*100:.2f}%")

# ============================================================
# STEP 3: FEATURE SELECTION & PREPROCESSING
# ============================================================

features = [
    "Movement_Speed", "Sound_Level_dB", "Crowd_Density",
    "Distance_Between_People_m", "Queue_Time_minutes",
    "Temperature", "Stress_Level", "Fatigue_Level",
    "Waiting_Time_for_Transport",
    "Security_Checkpoint_Wait_Time",
    "Interaction_Frequency",
    "Time_Spent_at_Location_minutes"
]

features = [f for f in features if f in df.columns]

X = df[features].copy()

# Map categorical columns to numeric
category_mapping = {"Low": 0.33, "Medium": 0.67, "High": 1.0}
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].map(category_mapping)

X = X.fillna(X.mean(numeric_only=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Feature matrix prepared: {X_scaled.shape}")

# ============================================================
# STEP 4: ISOLATION FOREST (UNSUPERVISED TRAINING)
# ============================================================

# Contamination slightly higher than proxy anomaly rate
contamination = min(max(actual_anomaly_rate * 1.2, 0.10), 0.25)

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=contamination,
    random_state=42,
    n_jobs=-1
)

# Unsupervised fit
pred = iso_forest.fit_predict(X_scaled)
scores = iso_forest.score_samples(X_scaled)

df["Predicted_Anomaly"] = (pred == -1).astype(int)
df["Anomaly_Score"] = scores

print("Isolation Forest training & inference complete")

# ============================================================
# STEP 5: EVALUATION METRICS
# ============================================================

tn, fp, fn, tp = confusion_matrix(df["Is_Abnormal"], df["Predicted_Anomaly"]).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print("\nPerformance Metrics")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1-Score : {f1:.3f}")

# ============================================================
# STEP 6: DECISION TREE (RULE-BASED INCIDENT LOGIC)
# Explicit IF–THEN rules for security alerts
# ============================================================

def decision_tree_incident(row):
    """
    Rule-based decision tree for incident severity.
    """
    if row["Predicted_Anomaly"] == 1:
        if row["Crowd_Density"] > 0.9 and row["Sound_Level_dB"] > 85:
            return "CRITICAL"
        elif row["Movement_Speed"] < 0.3 and row["Distance_Between_People_m"] < 0.7:
            return "HIGH"
        elif row["Queue_Time_minutes"] > 40:
            return "MEDIUM"
        else:
            return "LOW"
    return "NORMAL"

df["Incident_Level"] = df.apply(decision_tree_incident, axis=1)

# ============================================================
# STEP 7: INCIDENT SUMMARY & SECURITY ACTION
# ============================================================

print("\nIncident Distribution:")
print(df["Incident_Level"].value_counts())

print("\nSecurity Response Policy:")
print("CRITICAL → Immediate dispatch")
print("HIGH     → Investigate within 5 minutes")
print("MEDIUM   → Monitor closely")
print("LOW      → Routine observation")

# ============================================================
# STEP 8: EXPORT RESULTS
# ============================================================

summary = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "TP", "FP", "TN", "FN"],
    "Value": [accuracy, precision, recall, f1, tp, fp, tn, fn]
})

summary.to_csv("anomaly_detection_results.csv", index=False)

df[df["Predicted_Anomaly"] == 1][
    ["Anomaly_Score", "Incident_Level"] + features[:5]
].head(100).to_csv("detected_anomalies_for_review.csv", index=False)

print("\nResults exported successfully.")
print("System ready for security team review.")
