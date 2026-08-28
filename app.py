import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(
    page_title="Breast Cancer Stage Predictor",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    padding: 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.main-header h1 {
    margin: 0;
    font-size: 2.3rem;
}

.main-header p {
    color: #dbeafe;
    margin-top: 8px;
}

.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    padding: 18px;
    border-radius: 12px;
    margin: 20px 0;
}

.warning-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    padding: 18px;
    border-radius: 12px;
    margin-top: 30px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 30px 0;
    margin-top: 40px;
    border-top: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🧬 Breast Cancer Stage Predictor</h1>
    <p>SEER-based XGBoost Machine Learning Research Demonstration</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## About the System")

    st.write(
        "This application demonstrates a trained XGBoost "
        "classification model for predicting breast cancer stage "
        "from clinical dataset records."
    )

    st.markdown("---")

    st.markdown("### Model Information")
    st.write("**Algorithm:** XGBoost")
    st.write("**Dataset:** SEER")
    st.write("**Task:** Multi-class classification")
    st.write("**Target:** 6th Stage")

    st.markdown("---")

    st.markdown("### Workflow")
    st.write("1. Upload XGBoost model")
    st.write("2. Upload target encoder")
    st.write("3. Upload feature encoders")
    st.write("4. Upload feature columns")
    st.write("5. Upload CSV dataset")
    st.write("6. Generate predictions")
    st.write("7. Evaluate results")
    st.write("8. Download results")

st.markdown("## Prediction Workspace")

st.markdown("""
<div class="info-box">
<strong>Required Files</strong><br><br>
1. xgboost_model.pkl<br>
2. target_encoder.pkl<br>
3. feature_encoders.pkl<br>
4. feature_columns.pkl<br>
5. Clinical CSV dataset
</div>
""", unsafe_allow_html=True)

st.markdown("### Model Files")

col1, col2 = st.columns(2)

with col1:
    model_file = st.file_uploader(
        "🧠 Upload XGBoost Model",
        type=["pkl"]
    )

with col2:
    target_encoder_file = st.file_uploader(
        "🎯 Upload Target Encoder",
        type=["pkl"]
    )

col3, col4 = st.columns(2)

with col3:
    feature_encoder_file = st.file_uploader(
        "🔤 Upload Feature Encoders",
        type=["pkl"]
    )

with col4:
    feature_columns_file = st.file_uploader(
        "📋 Upload Feature Columns",
        type=["pkl"]
    )

st.markdown("### Clinical Dataset")

dataset_file = st.file_uploader(
    "📊 Upload Clinical CSV Dataset",
    type=["csv"]
)

if dataset_file is not None:
    try:
        dataset_file.seek(0)

        preview_df = pd.read_csv(dataset_file)

        preview_df.columns = preview_df.columns.str.strip()

        st.markdown("### Dataset Preview")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Records",
                f"{len(preview_df):,}"
            )

        with col2:
            st.metric(
                "Columns",
                f"{len(preview_df.columns):,}"
            )

        st.dataframe(
            preview_df.head(10),
            use_container_width=True
        )

        dataset_file.seek(0)

    except Exception as error:
        st.error(
            f"Unable to read dataset: {error}"
        )


def run_prediction(
    model_file,
    target_encoder_file,
    feature_encoder_file,
    feature_columns_file,
    dataset_file
):

    model = joblib.load(model_file)

    target_encoder = joblib.load(
        target_encoder_file
    )

    feature_encoders = joblib.load(
        feature_encoder_file
    )

    feature_columns = joblib.load(
        feature_columns_file
    )

    if not isinstance(feature_columns, list):
        raise ValueError(
            "feature_columns.pkl must contain a list."
        )

    dataset_file.seek(0)

    df = pd.read_csv(dataset_file)

    if df.empty:
        raise ValueError(
            "The uploaded dataset is empty."
        )

    df.columns = df.columns.str.strip()

    empty_unnamed = [
        column
        for column in df.columns
        if column.lower().startswith("unnamed")
        and df[column].isna().all()
    ]

    if empty_unnamed:
        df = df.drop(
            columns=empty_unnamed
        )

    target_column = "6th Stage"

    has_target = target_column in df.columns

    if has_target:
        X = df.drop(
            columns=[target_column]
        ).copy()
    else:
        X = df.copy()

    missing_features = [
        column
        for column in feature_columns
        if column not in X.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    X = X[feature_columns].copy()

    for column in feature_columns:

        if column not in feature_encoders:
            raise ValueError(
                f"No encoder found for '{column}'."
            )

        encoder = feature_encoders[column]

        values = X[column].astype(str)

        known_values = set(
            encoder.classes_
        )

        unknown_values = (
            set(values) - known_values
        )

        if unknown_values:

            examples = list(
                unknown_values
            )[:10]

            raise ValueError(
                f"Unknown value(s) found in "
                f"'{column}': "
                + ", ".join(
                    map(str, examples)
                )
            )

        X[column] = encoder.transform(
            values
        )

    predictions_encoded = model.predict(
        X.values
    )

    predictions = target_encoder.inverse_transform(
        predictions_encoded.astype(int)
    )

    result_df = df.copy()

    result_df["Predicted Stage"] = predictions

    metrics = None

    if has_target:

        actual_encoded = target_encoder.transform(
            df[target_column].astype(str)
        )

        accuracy = accuracy_score(
            actual_encoded,
            predictions_encoded
        )

        precision = precision_score(
            actual_encoded,
            predictions_encoded,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            actual_encoded,
            predictions_encoded,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            actual_encoded,
            predictions_encoded,
            average="weighted",
            zero_division=0
        )

        cm = confusion_matrix(
            actual_encoded,
            predictions_encoded
        )

        correct = int(
            np.sum(
                actual_encoded == predictions_encoded
            )
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "correct": correct,
            "confusion_matrix": cm,
            "classes": target_encoder.classes_
        }

    return result_df, metrics


all_files_uploaded = all([
    model_file is not None,
    target_encoder_file is not None,
    feature_encoder_file is not None,
    feature_columns_file is not None,
    dataset_file is not None
])

st.markdown("---")

st.markdown("## Generate Prediction")

if all_files_uploaded:

    st.success(
        "All required files have been uploaded."
    )

    if st.button(
        "🚀 Run XGBoost Prediction",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Processing dataset and generating predictions..."
        ):

            try:

                result_df, metrics = run_prediction(
                    model_file,
                    target_encoder_file,
                    feature_encoder_file,
                    feature_columns_file,
                    dataset_file
                )

                st.session_state["result_df"] = result_df
                st.session_state["metrics"] = metrics

                st.success(
                    "Prediction completed successfully."
                )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )

else:

    st.info(
        "Upload all four model artifacts and the "
        "CSV dataset to enable prediction."
    )


if "result_df" in st.session_state:

    result_df = st.session_state["result_df"]

    metrics = st.session_state["metrics"]

    st.markdown("---")

    st.markdown("## Prediction Results")

    if metrics is not None:

        st.markdown("### Model Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Accuracy",
                f"{metrics['accuracy'] * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Precision",
                f"{metrics['precision'] * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Recall",
                f"{metrics['recall'] * 100:.2f}%"
            )

        with col4:
            st.metric(
                "F1 Score",
                f"{metrics['f1'] * 100:.2f}%"
            )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Records Processed",
                f"{len(result_df):,}"
            )

        with col2:
            st.metric(
                "Correct Predictions",
                f"{metrics['correct']:,}"
            )

        st.markdown("### Confusion Matrix")

        cm = metrics["confusion_matrix"]

        classes = metrics["classes"]

        cm_df = pd.DataFrame(
            cm,
            index=[
                f"Actual: {item}"
                for item in classes
            ],
            columns=[
                f"Predicted: {item}"
                for item in classes
            ]
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )

    else:

        st.info(
            "Predictions were generated successfully, "
            "but the uploaded dataset does not contain "
            "the '6th Stage' column. Model evaluation "
            "metrics therefore cannot be calculated."
        )

    st.markdown("### Prediction Preview")

    st.dataframe(
        result_df.head(50),
        use_container_width=True
    )

    csv_data = result_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction Results",
        data=csv_data,
        file_name="breast_cancer_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )


st.markdown("""
<div class="warning-box">
<strong>Research and Educational Use Only</strong><br><br>
This application is a machine learning research demonstration.
Predictions should not be interpreted as a medical diagnosis,
prognosis, or treatment recommendation. Clinical decisions
should be made by qualified healthcare professionals using
appropriate clinical evidence.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
<strong>Breast Cancer Stage Predictor</strong><br>
XGBoost Machine Learning Research Demonstration
</div>
""", unsafe_allow_html=True)
