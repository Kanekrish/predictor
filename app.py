import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix
)

# ============================================================

# PAGE CONFIGURATION

# ============================================================

st.set_page_config(
page_title="Breast Cancer Stage Predictor",
page_icon="🧬",
layout="wide",
initial_sidebar_state="expanded"
)

# ============================================================

# CUSTOM CSS

# ============================================================

st.markdown(
""" <style>

```
.main-header {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
}

.main-header h1 {
    margin: 0;
    font-size: 2.4rem;
    font-weight: 700;
}

.main-header p {
    margin-top: 10px;
    color: #dbeafe;
    font-size: 1.05rem;
}

.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    padding: 20px;
    border-radius: 12px;
    margin: 20px 0;
    color: #1e3a8a;
}

.warning-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    padding: 20px;
    border-radius: 12px;
    margin-top: 30px;
    color: #92400e;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 30px 0 15px 0;
    border-top: 1px solid #e2e8f0;
    margin-top: 45px;
}

div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e2e8f0;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
}

</style>
""",
unsafe_allow_html=True
```

)

# ============================================================

# HEADER

# ============================================================

st.markdown(
""" <div class="main-header"> <h1>🧬 Breast Cancer Stage Predictor</h1> <p>
SEER-based XGBoost Machine Learning Research Demonstration </p> </div>
""",
unsafe_allow_html=True
)

# ============================================================

# SIDEBAR

# ============================================================

with st.sidebar:

```
st.markdown("## About the System")

st.write(
    """
    This application demonstrates the use of a trained
    XGBoost machine learning classification model to
    predict breast cancer stage from clinical dataset
    records.
    """
)

st.markdown("---")

st.markdown("### Model Information")

st.write("**Algorithm:** XGBoost")
st.write("**Dataset:** SEER")
st.write("**Task:** Multi-class classification")
st.write("**Target Variable:** 6th Stage")

st.markdown("---")

st.markdown("### Application Workflow")

st.write("1. Upload trained XGBoost model")
st.write("2. Upload target encoder")
st.write("3. Upload feature encoders")
st.write("4. Upload feature columns")
st.write("5. Upload clinical CSV dataset")
st.write("6. Generate predictions")
st.write("7. Evaluate model performance")
st.write("8. Download prediction results")

st.markdown("---")

st.markdown("### Research Notice")

st.write(
    """
    This system is intended for research and educational
    demonstration. It is not intended to replace clinical
    assessment or professional medical judgment.
    """
)
```

# ============================================================

# PREDICTION WORKSPACE

# ============================================================

st.markdown("## Prediction Workspace")

st.markdown(
"""
Upload the trained XGBoost model and the preprocessing
artifacts used during model training. Then upload a
compatible clinical CSV dataset.
"""
)

# ============================================================

# REQUIRED FILE INFORMATION

# ============================================================

st.markdown(
""" <div class="info-box"> <strong>Required Files</strong><br><br>
🧠 xgboost_model.pkl<br>
🎯 target_encoder.pkl<br>
🔤 feature_encoders.pkl<br>
📋 feature_columns.pkl<br>
📊 Clinical dataset (.csv) </div>
""",
unsafe_allow_html=True
)

# ============================================================

# MODEL FILE UPLOADS

# ============================================================

st.markdown("### 🧠 Model and Preprocessing Artifacts")

col1, col2 = st.columns(2)

with col1:

```
model_file = st.file_uploader(
    "Upload XGBoost Model",
    type=["pkl"],
    key="model_file",
    help="Upload the trained xgboost_model.pkl file."
)
```

with col2:

```
target_encoder_file = st.file_uploader(
    "Upload Target Encoder",
    type=["pkl"],
    key="target_encoder_file",
    help="Upload target_encoder.pkl."
)
```

col3, col4 = st.columns(2)

with col3:

```
feature_encoder_file = st.file_uploader(
    "Upload Feature Encoders",
    type=["pkl"],
    key="feature_encoder_file",
    help="Upload feature_encoders.pkl."
)
```

with col4:

```
feature_columns_file = st.file_uploader(
    "Upload Feature Columns",
    type=["pkl"],
    key="feature_columns_file",
    help="Upload feature_columns.pkl."
)
```

# ============================================================

# DATASET UPLOAD

# ============================================================

st.markdown("### 📊 Clinical Dataset")

dataset_file = st.file_uploader(
"Upload Clinical CSV Dataset",
type=["csv"],
key="dataset_file",
help="Upload the dataset that will be used for prediction."
)

# ============================================================

# DATASET PREVIEW

# ============================================================

if dataset_file is not None:

```
try:

    dataset_file.seek(0)

    preview_df = pd.read_csv(
        dataset_file
    )

    preview_df.columns = (
        preview_df.columns.str.strip()
    )

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
        f"Unable to read the uploaded CSV: {error}"
    )
```

# ============================================================

# PREDICTION FUNCTION

# ============================================================

def run_prediction(
model_file,
target_encoder_file,
feature_encoder_file,
feature_columns_file,
dataset_file
):

```
# --------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------

model = joblib.load(
    model_file
)


# --------------------------------------------------------
# LOAD TARGET ENCODER
# --------------------------------------------------------

target_encoder = joblib.load(
    target_encoder_file
)


# --------------------------------------------------------
# LOAD FEATURE ENCODERS
# --------------------------------------------------------

feature_encoders = joblib.load(
    feature_encoder_file
)


# --------------------------------------------------------
# LOAD FEATURE COLUMN LIST
# --------------------------------------------------------

feature_columns = joblib.load(
    feature_columns_file
)


if not isinstance(
    feature_columns,
    list
):

    raise ValueError(
        "feature_columns.pkl must contain a list of feature names."
    )


# --------------------------------------------------------
# LOAD DATASET
# --------------------------------------------------------

dataset_file.seek(0)

df = pd.read_csv(
    dataset_file
)


if df.empty:

    raise ValueError(
        "The uploaded dataset is empty."
    )


df.columns = (
    df.columns.str.strip()
)


# --------------------------------------------------------
# REMOVE COMPLETELY EMPTY UNNAMED COLUMNS
# --------------------------------------------------------

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


# --------------------------------------------------------
# TARGET COLUMN
# --------------------------------------------------------

target_column = "6th Stage"

has_target = (
    target_column in df.columns
)


# --------------------------------------------------------
# CREATE FEATURE DATAFRAME
# --------------------------------------------------------

if has_target:

    X = df.drop(
        columns=[target_column]
    ).copy()

else:

    X = df.copy()


# --------------------------------------------------------
# CHECK REQUIRED FEATURES
# --------------------------------------------------------

missing_features = [

    column

    for column in feature_columns

    if column not in X.columns

]


if missing_features:

    raise ValueError(
        "The uploaded dataset is missing "
        "the following required feature(s): "
        + ", ".join(missing_features)
    )


# --------------------------------------------------------
# KEEP TRAINING FEATURE ORDER
# --------------------------------------------------------

X = X[
    feature_columns
].copy()


# --------------------------------------------------------
# APPLY FEATURE ENCODERS
# --------------------------------------------------------

for column in feature_columns:

    if column not in feature_encoders:

        raise ValueError(
            f"No encoder found for feature '{column}'."
        )


    encoder = (
        feature_encoders[column]
    )


    values = (
        X[column].astype(str)
    )


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
            f"Unknown value(s) found "
            f"in feature '{column}': "
            + ", ".join(
                map(str, examples)
            )
        )


    X[column] = encoder.transform(
        values
    )


# --------------------------------------------------------
# GENERATE PREDICTIONS
# --------------------------------------------------------

predictions_encoded = (
    model.predict(
        X.values
    )
)


# --------------------------------------------------------
# CONVERT PREDICTIONS TO ORIGINAL STAGE LABELS
# --------------------------------------------------------

predictions = (
    target_encoder.inverse_transform(
        predictions_encoded.astype(int)
    )
)


# --------------------------------------------------------
# CREATE RESULTS DATAFRAME
# --------------------------------------------------------

result_df = df.copy()

result_df[
    "Predicted Stage"
] = predictions


# --------------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------------

metrics = None


if has_target:

    actual_values = (
        df[target_column].astype(str)
    )


    actual_encoded = (
        target_encoder.transform(
            actual_values
        )
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
            actual_encoded ==
            predictions_encoded
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
```

# ============================================================

# CHECK REQUIRED FILES

# ============================================================

all_files_uploaded = all(
[
model_file,
target_encoder_file,
feature_encoder_file,
feature_columns_file,
dataset_file
]
)

# ============================================================

# RUN PREDICTION

# ============================================================

st.markdown("---")

st.markdown("## Generate Prediction")

if all_files_uploaded:

```
st.success(
    "All required files have been uploaded and are ready for processing."
)


if st.button(
    "🚀 Run XGBoost Prediction",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Loading model and processing the dataset..."
    ):

        try:

            result_df, metrics = run_prediction(
                model_file,
                target_encoder_file,
                feature_encoder_file,
                feature_columns_file,
                dataset_file
            )


            st.session_state[
                "result_df"
            ] = result_df


            st.session_state[
                "metrics"
            ] = metrics


            st.success(
                "Prediction completed successfully."
            )


        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )
```

else:

```
st.info(
    """
    Please upload all four model artifacts and the
    clinical CSV dataset before running the prediction.
    """
)
```

# ============================================================

# DISPLAY RESULTS

# ============================================================

if "result_df" in st.session_state:

```
result_df = (
    st.session_state["result_df"]
)


metrics = (
    st.session_state["metrics"]
)


st.markdown("---")

st.markdown("## Prediction Results")


# ========================================================
# METRICS
# ========================================================

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


    # ====================================================
    # CONFUSION MATRIX
    # ====================================================

    st.markdown("### Confusion Matrix")


    cm = metrics[
        "confusion_matrix"
    ]


    class_names = metrics[
        "classes"
    ]


    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual: {name}"
            for name in class_names
        ],
        columns=[
            f"Predicted: {name}"
            for name in class_names
        ]
    )


    st.dataframe(
        cm_df,
        use_container_width=True
    )


else:

    st.info(
        """
        Predictions were generated successfully.

        However, the uploaded dataset does not contain
        the '6th Stage' target column. Therefore,
        accuracy, precision, recall, F1 score, and the
        confusion matrix cannot be calculated.
        """
    )


# ========================================================
# PREDICTION PREVIEW
# ========================================================

st.markdown("### Prediction Preview")


st.write(
    f"Showing the first 50 of "
    f"**{len(result_df):,} records**."
)


st.dataframe(
    result_df.head(50),
    use_container_width=True
)


# ========================================================
# DOWNLOAD RESULTS
# ========================================================

st.markdown("### Download Results")


csv_data = (
    result_df.to_csv(
        index=False
    ).encode("utf-8")
)


st.download_button(
    label="⬇️ Download Prediction Results",
    data=csv_data,
    file_name="breast_cancer_predictions.csv",
    mime="text/csv",
    use_container_width=True
)
```

# ============================================================

# DISCLAIMER

# ============================================================

st.markdown(
""" <div class="warning-box">

```
    <strong>Research and Educational Use Only</strong><br><br>

    This application is a machine learning research
    demonstration. Predictions generated by the model
    should not be interpreted as a medical diagnosis,
    prognosis, or treatment recommendation.

    Clinical decisions should be made by qualified
    healthcare professionals using appropriate clinical
    evidence.

</div>
""",
unsafe_allow_html=True
```

)

# ============================================================

# FOOTER

# ============================================================

st.markdown(
""" <div class="footer"> <strong>Breast Cancer Stage Predictor</strong> <br>
XGBoost Machine Learning Research Demonstration </div>
""",
unsafe_allow_html=True
)
