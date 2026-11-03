import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

print("===================================")
print(" DBSCAN CLUSTER VISUALIZATION")
print("===================================")

os.makedirs("model", exist_ok=True)

# Load dataset
data = pd.read_csv("dataset/UNSW_NB15_training-set.csv")

true_labels = data["label"].copy()
attack_categories = data["attack_cat"].copy()

# Remove unnecessary columns
features = data.drop(
    columns=["id", "attack_cat", "label"]
)

# Convert categorical data
features = pd.get_dummies(
    features,
    columns=["proto", "service", "state"],
    dtype=int
)

# Handle missing and infinite values
features = features.replace(
    [np.inf, -np.inf],
    np.nan
)

features = features.fillna(
    features.median()
)

# Standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

print("\nDataset Loaded Successfully!")
print("Total Records:", len(data))

# Select 5000 records
sample_size = min(5000, len(features))

sample_indices = features.sample(
    n=sample_size,
    random_state=42
).index.to_numpy()

sample_data = scaled_data[sample_indices]

print("\nSample Records:", sample_size)

# PCA dimensionality reduction
pca = PCA(
    n_components=10,
    random_state=42
)

reduced_data = pca.fit_transform(sample_data)

print("\nPCA Completed Successfully!")

# DBSCAN clustering
model = DBSCAN(
    eps=3.0,
    min_samples=10,
    n_jobs=-1
)

predictions = model.fit_predict(reduced_data)

# Calculate results
noise_count = int(np.sum(predictions == -1))

cluster_count = len(
    set(predictions)
) - (1 if -1 in predictions else 0)

print("\nDBSCAN RESULTS")
print("-----------------------------")
print("Total Records:", sample_size)
print("Clusters Found:", cluster_count)
print("Noise / Anomalies:", noise_count)

# Save models
joblib.dump(model, "model/dbscan.pkl")
joblib.dump(scaler, "model/dbscan_scaler.pkl")
joblib.dump(pca, "model/pca.pkl")

# Create visualization results
results = pd.DataFrame({
    "Actual_Label": true_labels.iloc[sample_indices].values,
    "Attack_Category": attack_categories.iloc[sample_indices].values,
    "DBSCAN_Result": predictions,
    "PCA_X": reduced_data[:, 0],
    "PCA_Y": reduced_data[:, 1]
})

results["Status"] = results["DBSCAN_Result"].apply(
    lambda x: "Noise / Anomaly"
    if x == -1
    else "Cluster " + str(x)
)

# Save CSV
results.to_csv(
    "model/dbscan_results.csv",
    index=False
)

print("\nModels Saved Successfully!")
print("Visualization Data Saved Successfully!")

print("\nSTEP 11 COMPLETED!")
