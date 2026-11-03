from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

# Allow CSV uploads up to 100 MB
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


# ==========================================
# DASHBOARD DATA
# ==========================================

def get_dashboard_data():

    if_results = pd.read_csv(
        MODEL_DIR / "isolation_forest_results.csv"
    )

    db_data = pd.read_csv(
        MODEL_DIR / "dbscan_results.csv"
    )

    comparison_file = MODEL_DIR / "model_comparison.csv"

    if comparison_file.exists():
        comparison_data = pd.read_csv(comparison_file)
    else:
        comparison_data = pd.DataFrame()

    total_records = len(if_results)

    normal_count = int(
        (if_results["Status"] == "Normal").sum()
    )

    anomaly_count = int(
        (if_results["Status"] == "Anomaly").sum()
    )

    anomaly_rate = (
        round((anomaly_count / total_records) * 100, 2)
        if total_records > 0 else 0
    )

    attack_data = if_results["Attack_Category"].value_counts()

    attack_labels = attack_data.index.tolist()
    attack_counts = attack_data.values.tolist()

    anomaly_records = if_results[
        if_results["Status"] == "Anomaly"
    ].head(100).to_dict(orient="records")

    if {"PCA_X", "PCA_Y", "DBSCAN_Result"}.issubset(
        db_data.columns
    ):

        dbscan_visual = json.loads(
            db_data[
                ["PCA_X", "PCA_Y", "DBSCAN_Result"]
            ].to_json(orient="records")
        )

    else:
        dbscan_visual = []

    comparison_columns = comparison_data.columns.tolist()

    comparison_rows = json.loads(
        comparison_data.to_json(orient="records")
    )

    return {
        "total_records": total_records,
        "normal_count": normal_count,
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
        "attack_labels": attack_labels,
        "attack_counts": attack_counts,
        "anomaly_records": anomaly_records,
        "comparison_columns": comparison_columns,
        "comparison_rows": comparison_rows,
        "dbscan_visual": dbscan_visual,
        "dbscan_total": len(db_data),
        "dbscan_noise": int(
            (db_data["DBSCAN_Result"] == -1).sum()
        )
    }


# ==========================================
# MAIN DASHBOARD
# ==========================================

@app.route("/")
def dashboard():

    data = get_dashboard_data()

    data["cluster_json"] = data["dbscan_visual"]

    return render_template(
        "index.html",
        data=data
    )


# ==========================================
# ANOMALY DETECTION
# ==========================================

@app.route("/detect", methods=["GET", "POST"])
def detect():

    results = None
    normal_count = 0
    anomaly_count = 0
    total_records = 0
    error = None

    if request.method == "POST":

        file = request.files.get("file")

        if not file or file.filename == "":
            error = "Please select a CSV file."

        elif not file.filename.lower().endswith(".csv"):
            error = "Please upload a valid CSV file."

        else:

            try:

                # Load saved model files once
                model = joblib.load(
                    MODEL_DIR / "isolation_forest.pkl"
                )

                scaler = joblib.load(
                    MODEL_DIR / "scaler.pkl"
                )

                saved_features = joblib.load(
                    MODEL_DIR / "features.pkl"
                )

                result_file = (
                    MODEL_DIR / "uploaded_detection_results.csv"
                )

                total_records = 0
                normal_count = 0
                anomaly_count = 0

                # Process CSV in batches
                with open(result_file, "w", newline="") as output:

                    first_batch = True

                    for uploaded_data in pd.read_csv(
                        file,
                        chunksize=5000
                    ):

                        # Remove unnecessary columns
                        features = uploaded_data.drop(
                            columns=[
                                "id",
                                "attack_cat",
                                "label"
                            ],
                            errors="ignore"
                        )

                        # Convert categorical columns
                        categorical_columns = [
                            col for col in [
                                "proto",
                                "service",
                                "state"
                            ]
                            if col in features.columns
                        ]

                        features = pd.get_dummies(
                            features,
                            columns=categorical_columns,
                            dtype=int
                        )

                        # Match training features
                        features = features.reindex(
                            columns=saved_features,
                            fill_value=0
                        )

                        features = features.replace(
                            [np.inf, -np.inf],
                            np.nan
                        )

                        features = features.fillna(0)

                        # Scale current batch
                        scaled_data = scaler.transform(features)

                        # Predict anomalies
                        predictions = model.predict(scaled_data)

                        batch_results = pd.DataFrame({

                            "Record_Number": range(
                                total_records + 1,
                                total_records + len(predictions) + 1
                            ),

                            "Prediction": predictions,

                            "Status": [
                                "Normal" if x == 1 else "Anomaly"
                                for x in predictions
                            ]

                        })

                        # Update counts
                        batch_normal = int(
                            (batch_results["Status"] == "Normal").sum()
                        )

                        batch_anomaly = int(
                            (batch_results["Status"] == "Anomaly").sum()
                        )

                        normal_count += batch_normal
                        anomaly_count += batch_anomaly
                        total_records += len(batch_results)

                        # Save each batch immediately
                        batch_results.to_csv(
                            output,
                            index=False,
                            header=first_batch
                        )

                        first_batch = False

                        del uploaded_data
                        del features
                        del scaled_data
                        del predictions
                        del batch_results

                # Load only a small preview for the result page
                results = pd.read_csv(
                    result_file,
                    nrows=100
                )

            except Exception as e:

                error = "Detection failed: " + str(e)

    return render_template(
        "detect.html",
        results=results,
        normal_count=normal_count,
        anomaly_count=anomaly_count,
        total_records=total_records,
        error=error
    )


# ==========================================
# FILE SIZE ERROR
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    return (
        "File is too large. Please upload a CSV file smaller than 100 MB.",
        413
    )


# ==========================================
# DOWNLOAD DETECTION RESULTS
# ==========================================

@app.route("/download")
def download():

    file_path = MODEL_DIR / "uploaded_detection_results.csv"

    if file_path.exists():

        return send_file(
            file_path,
            as_attachment=True,
            download_name="Detection_Results.csv"
        )

    return "No detection results available."


# ==========================================
# PDF REPORT GENERATION
# ==========================================

@app.route("/report")
def report():

    data = get_dashboard_data()

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "NETWORK INTRUSION DETECTION SYSTEM",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "Project Report",
            styles["Heading2"]
        )
    )

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph(
            "Dataset: UNSW-NB15",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph(
            "Dataset Summary",
            styles["Heading2"]
        )
    )

    summary = [
        ["Description", "Value"],
        ["Total Records", str(data["total_records"])],
        ["Normal Activities", str(data["normal_count"])],
        ["Detected Anomalies", str(data["anomaly_count"])],
        ["Anomaly Rate", str(data["anomaly_rate"]) + "%"],
        ["DBSCAN Sample Records", str(data["dbscan_total"])],
        ["DBSCAN Noise Records", str(data["dbscan_noise"])]
    ]

    table = Table(summary, colWidths=[250, 150])

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("PADDING", (0, 0), (-1, -1), 8)
        ])
    )

    elements.append(table)

    elements.append(Spacer(1, 25))

    elements.append(
        Paragraph(
            "Conclusion",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            "The system uses Isolation Forest and DBSCAN "
            "to identify unusual network activities. "
            "PCA is used to reduce the dimensions of the "
            "sampled dataset for cluster visualization. "
            "Detected anomalies are model predictions "
            "and should not be treated as confirmed attacks.",
            styles["Normal"]
        )
    )

    document.build(elements)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Network_Security_Report.pdf",
        mimetype="application/pdf"
    )


# ==========================================
# RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=False
    )
