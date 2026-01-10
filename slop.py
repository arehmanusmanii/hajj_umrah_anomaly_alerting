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

# Map categorical columns to numeric first
category_mapping = {"Low": 0.33, "Medium": 0.67, "High": 1.0}
if "Crowd_Density" in df.columns:
    df["Crowd_Density"] = df["Crowd_Density"].map(category_mapping)
if "Stress_Level" in df.columns:
    df["Stress_Level"] = df["Stress_Level"].map(category_mapping)
if "Fatigue_Level" in df.columns:
    df["Fatigue_Level"] = df["Fatigue_Level"].map(category_mapping)

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

# Handle any remaining categorical columns
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
# STEP 8: DECISION-FOCUSED VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 7))

# Add small jitter to make overlapping points visible (Crowd_Density only has 3 values)
np.random.seed(42)
jitter_x = 0.02
jitter_y = 0.5

# Define color map and marker properties for incident levels
incident_colors = {
    "NORMAL": "green",
    "LOW": "#FFD700",        # Gold
    "MEDIUM": "#FFA500",     # Orange
    "HIGH": "#FF6347",       # Tomato red
    "CRITICAL": "#8B0000"    # Dark red
}

incident_sizes = {
    "NORMAL": 20,
    "LOW": 35,
    "MEDIUM": 50,
    "HIGH": 65,
    "CRITICAL": 80
}

incident_alphas = {
    "NORMAL": 0.3,
    "LOW": 0.5,
    "MEDIUM": 0.65,
    "HIGH": 0.8,
    "CRITICAL": 0.95
}

# Plot by incident level
for incident_level in ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]:
    subset = df[df["Incident_Level"] == incident_level].copy()
    if len(subset) > 0:
        subset_x = subset["Crowd_Density"] + np.random.normal(0, jitter_x, len(subset))
        subset_y = subset["Sound_Level_dB"] + np.random.normal(0, jitter_y, len(subset))
        
        ax.scatter(subset_x, subset_y,
                   c=incident_colors[incident_level],
                   alpha=incident_alphas[incident_level],
                   s=incident_sizes[incident_level],
                   label=f'{incident_level} ({len(subset)})',
                   edgecolors='black' if incident_level != "NORMAL" else 'none',
                   linewidth=1 if incident_level != "NORMAL" else 0)

# Draw decision thresholds
ax.axvline(x=0.9, color='blue', linestyle='--', linewidth=2, label='Density Threshold (0.9)', alpha=0.8)
ax.axhline(y=85, color='purple', linestyle='--', linewidth=2, label='Sound Threshold (85 dB)', alpha=0.8)

# Highlight critical zone (top-right quadrant)
density_min, density_max = df["Crowd_Density"].min(), df["Crowd_Density"].max()
sound_min, sound_max = df["Sound_Level_dB"].min(), df["Sound_Level_dB"].max()
ax.fill_between([0.9, density_max + 0.05], 85, sound_max + 2, alpha=0.15, color='red', label='Critical Zone')

ax.set_xlabel('Crowd Density (categorical: Low=0.33, Medium=0.67, High=1.0)', fontsize=11, fontweight='bold')
ax.set_ylabel('Sound Level (dB)', fontsize=11, fontweight='bold')
ax.set_title('Dispatch Decision Map: Incident Severity by Crowd Density & Sound Level', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.2)

# Dynamic axis limits with padding
ax.set_xlim(density_min - 0.1, density_max + 0.1)
ax.set_ylim(sound_min - 5, sound_max + 5)

plt.tight_layout()
plt.savefig('dispatch_decision_map.png', dpi=150, bbox_inches='tight')
print("Decision map saved as 'dispatch_decision_map.png'")
plt.close()

# ============================================================
# STEP 9: EXPORT RESULTS
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
