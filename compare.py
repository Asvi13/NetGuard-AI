import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("MODEL PERFORMANCE COMPARISON")

# Load Isolation Forest results
if_data = pd.read_csv(
    "model/isolation_forest_results.csv"
)

# Load DBSCAN results
db_data = pd.read_csv(
    "model/dbscan_results.csv"
)

# Actual labels
y_true_if = if_data["Actual_Label"]

# Convert Isolation Forest predictions
y_pred_if = if_data["Predicted_Result"].map({
    1: 0,
    -1: 1
})

# Actual DBSCAN labels
y_true_db = db_data["Actual_Label"]

# Convert DBSCAN predictions
y_pred_db = (db_data["DBSCAN_Result"] == -1).astype(int)


# Function to calculate performance
def evaluate_model(name, actual, predicted):

    print("\n------------------------------")
    print(name)
    print("------------------------------")

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual, predicted, zero_division=0
    )

    recall = recall_score(
        actual, predicted, zero_division=0
    )

    f1 = f1_score(
        actual, predicted, zero_division=0
    )

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F1 Score:", round(f1 * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(actual, predicted))

    return {
        "Algorithm": name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2)
    }


# Evaluate both algorithms
result1 = evaluate_model(
    "Isolation Forest",
    y_true_if,
    y_pred_if
)

result2 = evaluate_model(
    "DBSCAN",
    y_true_db,
    y_pred_db
)

# Comparison table
comparison = pd.DataFrame([
    result1,
    result2
])

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(index=False))

# Save comparison
comparison.to_csv(
    "model/model_comparison.csv",
    index=False
)

print("\nComparison Results Saved Successfully!")
