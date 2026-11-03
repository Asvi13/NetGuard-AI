import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

print("ISOLATION FOREST MODEL STARTED")

# Load dataset
data = pd.read_csv("dataset/UNSW_NB15_training-set.csv")

# Save actual labels separately for evaluation
true_labels = data["label"].copy()
attack_categories = data["attack_cat"].copy()

# Remove unnecessary columns
features = data.drop(columns=["id", "attack_cat", "label"])

# Convert categorical columns
features = pd.get_dummies(
    features,
    columns=["proto", "service", "state"],
    dtype=int
)

# Handle missing and infinite values
features = features.replace([np.inf, -np.inf], np.nan)
features = features.fillna(features.median())

# Feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

print("\nDataset Prepared Successfully!")

# Create Isolation Forest model
model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

# Train the model without labels
print("\nTraining Isolation Forest Model...")

model.fit(scaled_data)

# Predict anomalies
predictions = model.predict(scaled_data)

# Convert predictions
# 1 = Normal
# -1 = Anomaly

normal_count = np.sum(predictions == 1)
anomaly_count = np.sum(predictions == -1)

print("\nMODEL TRAINING COMPLETED!")

print("Total Records:", len(predictions))
print("Normal Activities:", normal_count)
print("Anomalies Detected:", anomaly_count)

# Save model and scaler
joblib.dump(model, "model/isolation_forest.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(features.columns.tolist(), "model/features.pkl")

# Save detection results
results = pd.DataFrame({
    "Actual_Label": true_labels,
    "Attack_Category": attack_categories,
    "Predicted_Result": predictions
})

results["Status"] = results["Predicted_Result"].map({
    1: "Normal",
    -1: "Anomaly"
})

results.to_csv("model/isolation_forest_results.csv", index=False)

print("\nModel Saved Successfully!")
print("Results Saved Successfully!")
