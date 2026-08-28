import io
import json
import base64
import joblib
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

app = FastAPI(
    title="Breast Cancer Stage Predictor",
    description="XGBoost-based breast cancer stage prediction system",
    version="1.0.0"
)


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Breast Cancer Stage Predictor</title>

        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">

        <style>

            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                font-family: Arial, Helvetica, sans-serif;
                background: #f4f7fb;
                color: #1f2937;
            }

            .header {
                background: #111827;
                color: white;
                padding: 30px 20px;
                text-align: center;
            }

            .header h1 {
                margin: 0 0 10px;
                font-size: 32px;
            }

            .header p {
                margin: 0;
                color: #d1d5db;
            }

            .container {
                max-width: 1000px;
                margin: 40px auto;
                padding: 0 20px;
            }

            .card {
                background: white;
                border-radius: 14px;
                padding: 30px;
                margin-bottom: 25px;
                box-shadow: 0 4px 18px rgba(0,0,0,0.08);
            }

            .card h2 {
                margin-top: 0;
                color: #111827;
            }

            .upload-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
            }

            .upload-box {
                border: 2px dashed #cbd5e1;
                border-radius: 10px;
                padding: 20px;
                background: #f8fafc;
            }

            .upload-box label {
                display: block;
                font-weight: bold;
                margin-bottom: 10px;
            }

            input[type="file"] {
                width: 100%;
            }

            .button-container {
                text-align: center;
                margin-top: 30px;
            }

            button {
                border: none;
                background: #2563eb;
                color: white;
                padding: 14px 30px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
            }

            button:hover {
                background: #1d4ed8;
            }

            button:disabled {
                background: #94a3b8;
                cursor: not-allowed;
            }

            #status {
                margin-top: 20px;
                padding: 15px;
                border-radius: 8px;
                display: none;
            }

            .success {
                background: #dcfce7;
                color: #166534;
            }

            .error {
                background: #fee2e2;
                color: #991b1b;
            }

            .loading {
                background: #e0f2fe;
                color: #075985;
            }

            .metrics {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin-top: 20px;
            }

            .metric {
                background: #f8fafc;
                padding: 20px;
                text-align: center;
                border-radius: 10px;
            }

            .metric h3 {
                margin: 0;
                font-size: 28px;
                color: #2563eb;
            }

            .metric p {
                margin: 8px 0 0;
                color: #64748b;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }

            th, td {
                padding: 10px;
                border-bottom: 1px solid #e5e7eb;
                text-align: left;
            }

            th {
                background: #f8fafc;
            }

            .download {
                display: inline-block;
                margin-top: 20px;
                padding: 12px 20px;
                background: #059669;
                color: white;
                text-decoration: none;
                border-radius: 7px;
            }

            .instructions {
                line-height: 1.7;
            }

            @media(max-width: 700px) {

                .upload-grid {
                    grid-template-columns: 1fr;
                }

                .metrics {
                    grid-template-columns: repeat(2, 1fr);
                }

            }

        </style>
    </head>

    <body>

        <div class="header">
            <h1>Breast Cancer Stage Predictor</h1>
            <p>XGBoost Machine Learning Prediction System</p>
        </div>

        <div class="container">

            <div class="card">

                <h2>About the System</h2>

                <p class="instructions">
                    This application uses a trained XGBoost machine learning
                    model to predict breast cancer stage from a SEER clinical
                    dataset.
                </p>

                <p class="instructions">
                    Upload the trained model artifacts and the dataset below
                    to perform predictions.
                </p>

            </div>


            <div class="card">

                <h2>Upload Model and Dataset</h2>

                <form id="predictionForm">

                    <div class="upload-grid">

                        <div class="upload-box">

                            <label>
                                XGBoost Model (.pkl)
                            </label>

                            <input
                                type="file"
                                name="model_file"
                                accept=".pkl"
                                required
                            >

                        </div>


                        <div class="upload-box">

                            <label>
                                Target Encoder (.pkl)
                            </label>

                            <input
                                type="file"
                                name="target_encoder"
                                accept=".pkl"
                                required
                            >

                        </div>


                        <div class="upload-box">

                            <label>
                                Feature Encoders (.pkl)
                            </label>

                            <input
                                type="file"
                                name="feature_encoders"
                                accept=".pkl"
                                required
                            >

                        </div>


                        <div class="upload-box">

                            <label>
                                Feature Columns (.pkl)
                            </label>

                            <input
                                type="file"
                                name="feature_columns"
                                accept=".pkl"
                                required
                            >

                        </div>


                        <div class="upload-box">

                            <label>
                                Dataset (.CSV)
                            </label>

                            <input
                                type="file"
                                name="dataset"
                                accept=".csv"
                                required
                            >

                        </div>

                    </div>


                    <div class="button-container">

                        <button
                            type="submit"
                            id="predictButton"
                        >
                            Run Prediction
                        </button>

                    </div>

                </form>


                <div id="status"></div>

            </div>


            <div
                class="card"
                id="results"
                style="display:none;"
            >

                <h2>Prediction Results</h2>

                <div
                    class="metrics"
                    id="metrics"
                >
                </div>

                <div id="summary"></div>

                <div id="tableContainer"></div>

            </div>

        </div>


        <script>

            const form =
                document.getElementById("predictionForm");

            const status =
                document.getElementById("status");

            const results =
                document.getElementById("results");

            const metrics =
                document.getElementById("metrics");

            const summary =
                document.getElementById("summary");

            const tableContainer =
                document.getElementById("tableContainer");


            form.addEventListener("submit", async function(event) {

                event.preventDefault();

                const button =
                    document.getElementById("predictButton");

                button.disabled = true;

                status.style.display = "block";
                status.className = "loading";
                status.innerText =
                    "Processing dataset and generating predictions...";

                results.style.display = "none";


                const formData =
                    new FormData(form);


                try {

                    const response =
                        await fetch(
                            "/predict",
                            {
                                method: "POST",
                                body: formData
                            }
                        );


                    const data =
                        await response.json();


                    if (!response.ok) {

                        throw new Error(
                            data.detail ||
                            "Prediction failed."
                        );

                    }


                    status.className = "success";

                    status.innerText =
                        "Prediction completed successfully.";


                    results.style.display = "block";


                    metrics.innerHTML = `

                        <div class="metric">

                            <h3>
                                ${data.total_records}
                            </h3>

                            <p>Total Records</p>

                        </div>


                        <div class="metric">

                            <h3>
                                ${data.correct_predictions ?? "N/A"}
                            </h3>

                            <p>Correct Predictions</p>

                        </div>


                        <div class="metric">

                            <h3>
                                ${data.accuracy
                                    ? (data.accuracy * 100).toFixed(2) + "%"
                                    : "N/A"}
                            </h3>

                            <p>Accuracy</p>

                        </div>


                        <div class="metric">

                            <h3>
                                ${data.f1_score
                                    ? (data.f1_score * 100).toFixed(2) + "%"
                                    : "N/A"}
                            </h3>

                            <p>F1 Score</p>

                        </div>

                    `;


                    summary.innerHTML = `

                        <p>
                            <strong>Prediction column:</strong>
                            ${data.prediction_column}
                        </p>

                        <p>
                            <strong>Dataset rows:</strong>
                            ${data.total_records}
                        </p>

                    `;


                    let html =
                        "<table><thead><tr>";


                    data.preview_columns.forEach(
                        column => {

                            html +=
                                `<th>${column}</th>`;

                        }
                    );


                    html +=
                        "</tr></thead><tbody>";


                    data.preview_rows.forEach(
                        row => {

                            html += "<tr>";

                            data.preview_columns.forEach(
                                column => {

                                    html +=
                                        `<td>${row[column]}</td>`;

                                }
                            );

                            html += "</tr>";

                        }
                    );


                    html +=
                        "</tbody></table>";


                    tableContainer.innerHTML =
                        html;


                    if (data.download_data) {

                        const link =
                            document.createElement("a");

                        link.className = "download";

                        link.href =
                            "data:text/csv;base64," +
                            data.download_data;

                        link.download =
                            "breast_cancer_predictions.csv";

                        link.innerText =
                            "Download Prediction Results";

                        tableContainer.appendChild(link);

                    }

                }

                catch(error) {

                    status.className = "error";

                    status.innerText =
                        error.message;

                }

                finally {

                    button.disabled = false;

                }

            });

        </script>

    </body>
    </html>
    """


@app.post("/predict")
async def predict(
    model_file: UploadFile = File(...),
    target_encoder: UploadFile = File(...),
    feature_encoders: UploadFile = File(...),
    feature_columns: UploadFile = File(...),
    dataset: UploadFile = File(...)
):

    try:

        # -------------------------------------------------
        # Validate uploaded files
        # -------------------------------------------------

        if not model_file.filename.endswith(".pkl"):
            raise HTTPException(
                status_code=400,
                detail="Model file must be a .pkl file."
            )

        if not target_encoder.filename.endswith(".pkl"):
            raise HTTPException(
                status_code=400,
                detail="Target encoder must be a .pkl file."
            )

        if not feature_encoders.filename.endswith(".pkl"):
            raise HTTPException(
                status_code=400,
                detail="Feature encoders file must be a .pkl file."
            )

        if not feature_columns.filename.endswith(".pkl"):
            raise HTTPException(
                status_code=400,
                detail="Feature columns file must be a .pkl file."
            )

        if not dataset.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Dataset must be a CSV file."
            )


        # -------------------------------------------------
        # Load uploaded artifacts
        # -------------------------------------------------

        model =
            joblib.load(
                io.BytesIO(
                    await model_file.read()
                )
            )

        target_enc =
            joblib.load(
                io.BytesIO(
                    await target_encoder.read()
                )
            )

        feature_enc =
            joblib.load(
                io.BytesIO(
                    await feature_encoders.read()
                )
            )

        feature_cols =
            joblib.load(
                io.BytesIO(
                    await feature_columns.read()
                )
            )


        # -------------------------------------------------
        # Read dataset
        # -------------------------------------------------

        dataset_bytes =
            await dataset.read()

        df =
            pd.read_csv(
                io.BytesIO(dataset_bytes)
            )

        df.columns =
            df.columns.str.strip()


        if df.empty:

            raise HTTPException(
                status_code=400,
                detail="The uploaded dataset is empty."
            )


        # -------------------------------------------------
        # Remove completely empty unnamed columns
        # -------------------------------------------------

        empty_unnamed = [

            c for c in df.columns

            if c.lower().startswith("unnamed")
            and df[c].isna().all()

        ]

        if empty_unnamed:

            df =
                df.drop(
                    columns=empty_unnamed
                )


        # -------------------------------------------------
        # Identify target column
        # -------------------------------------------------

        target_column = "6th Stage"

        has_target =
            target_column in df.columns


        # -------------------------------------------------
        # Create prediction dataset
        # -------------------------------------------------

        if has_target:

            X =
                df.drop(
                    columns=[target_column]
                ).copy()

        else:

            X =
                df.copy()


        # -------------------------------------------------
        # Ensure expected features exist
        # -------------------------------------------------

        missing_features = [

            col for col in feature_cols
            if col not in X.columns

        ]

        if missing_features:

            raise HTTPException(

                status_code=400,

                detail=
                    "Missing required features: "
                    +
                    ", ".join(missing_features)

            )


        # Keep only features used by model

        X =
            X[
                feature_cols
            ].copy()


        # -------------------------------------------------
        # Apply training-time feature encoders
        # -------------------------------------------------

        for col in feature_cols:

            if col not in feature_enc:

                raise HTTPException(

                    status_code=400,

                    detail=
                        f"No encoder found for feature '{col}'."

                )


            encoder =
                feature_enc[col]


            values =
                X[col].astype(str)


            known_values =
                set(
                    encoder.classes_
                )


            unknown_values =
                set(values) - known_values


            if unknown_values:

                raise HTTPException(

                    status_code=400,

                    detail=
                        f"Unknown value(s) found in "
                        f"feature '{col}': "
                        +
                        ", ".join(
                            list(unknown_values)[:10]
                        )

                )


            X[col] =
                encoder.transform(
                    values
                )


        # -------------------------------------------------
        # Generate predictions
        # -------------------------------------------------

        predictions_encoded =
            model.predict(
                X.values
            )


        predictions =
            target_enc.inverse_transform(
                predictions_encoded.astype(int)
            )


        # -------------------------------------------------
        # Add predictions
        # -------------------------------------------------

        result_df =
            df.copy()


        result_df[
            "Predicted Stage"
        ] =
            predictions


        # -------------------------------------------------
        # Calculate metrics if target exists
        # -------------------------------------------------

        accuracy = None
        precision = None
        recall = None
        f1 = None
        correct_predictions = None


        if has_target:

            actual =
                target_enc.transform(
                    df[target_column].astype(str)
                )


            correct_predictions =
                int(
                    (actual == predictions_encoded).sum()
                )


            accuracy =
                accuracy_score(
                    actual,
                    predictions_encoded
                )


            precision =
                precision_score(
                    actual,
                    predictions_encoded,
                    average="weighted",
                    zero_division=0
                )


            recall =
                recall_score(
                    actual,
                    predictions_encoded,
                    average="weighted",
                    zero_division=0
                )


            f1 =
                f1_score(
                    actual,
                    predictions_encoded,
                    average="weighted",
                    zero_division=0
                )


        # -------------------------------------------------
        # Prepare downloadable CSV
        # -------------------------------------------------

        csv_buffer =
            io.StringIO()


        result_df.to_csv(
            csv_buffer,
            index=False
        )


        encoded_csv =
            base64.b64encode(
                csv_buffer.getvalue().encode()
            ).decode()


        # -------------------------------------------------
        # Preview
        # -------------------------------------------------

        preview_df =
            result_df.head(20).copy()


        preview_columns =
            list(
                preview_df.columns
            )


        preview_rows =
            json.loads(
                preview_df.to_json(
                    orient="records"
                )
            )


        # -------------------------------------------------
        # Return results
        # -------------------------------------------------

        return {

            "success": True,

            "total_records":
                len(result_df),

            "correct_predictions":
                correct_predictions,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1_score":
                f1,

            "prediction_column":
                "Predicted Stage",

            "preview_columns":
                preview_columns,

            "preview_rows":
                preview_rows,

            "download_data":
                encoded_csv

        }


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=
                f"Prediction error: {str(e)}"

        )
