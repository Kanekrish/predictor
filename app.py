
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

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
/* Main page */

.main {
    background-color: #f8fafc;
}

/* Header */

.main-header {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e3a8a 100%
    );

    padding: 2.2rem 2.5rem;

    border-radius: 18px;

    color: white;

    margin-bottom: 2rem;

    box-shadow:
        0 10px 30px rgba(15, 23, 42, 0.12);
}

.main-header h1 {
    font-size: 2.35rem;
    margin-bottom: 0.4rem;
    font-weight: 750;
}

.main-header p {
    color: #dbeafe;
    font-size: 1rem;
    margin-bottom: 0;
}

/* Section headings */

.section-title {
    font-size: 1.35rem;
    font-weight: 700;

    color: #0f172a;

    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}

.section-description {
    color: #64748b;
    margin-bottom: 1.2rem;
}

/* Information boxes */

.info-box {
    padding: 1rem 1.2rem;

    border-radius: 12px;

    background: #eff6ff;

    border: 1px solid #bfdbfe;

    color: #1e3a8a;

    margin-bottom: 1.5rem;
}

.warning-box {
    padding: 1rem 1.2rem;

    border-radius: 12px;

    background: #fffbeb;

    border: 1px solid #fde68a;

    color: #92400e;

    margin-top: 1.5rem;
}

/* Upload cards */

.upload-card {
    background: white;

    padding: 1.3rem;

    border-radius: 14px;

    border: 1px solid #e2e8f0;

    margin-bottom: 1rem;

    box-shadow:
        0 4px 14px rgba(15, 23, 42, 0.04);
}

/* Metric cards */

div[data-testid="metric-container"] {

    background: white;

    border: 1px solid #e2e8f0;

    padding: 1rem;

    border-radius: 12px;

    box-shadow:
        0 4px 14px rgba(15, 23, 42, 0.04);

}

/* Buttons */

.stButton > button {

    width: 100%;

    border-radius: 9px;

    height: 3rem;

    font-weight: 700;

}

/* Sidebar */

section[data-testid="stSidebar"] {

    background-color: #f8fafc;

    border-right: 1px solid #e2e8f0;

}

/* Footer */

.footer {

    text-align: center;

    color: #64748b;

    font-size: 0.8rem;

    padding: 2rem 0 1rem;

    border-top: 1px solid #e2e8f0;

    margin-top: 3rem;

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
""" <div class="main-header">

```
    <h1>
        🧬 Breast Cancer Stage Predictor
    </h1>

    <p>
        SEER-based XGBoost Machine Learning Research Demonstration
    </p>

</div>
""",
unsafe_allow_html=True
```

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
    XGBoost classification model to predict breast cancer
    stage from clinical dataset records.
    """
)

st.markdown("---")

st.markdown("### Model Information")

st.write("**Algorithm:** XGBoost")
st.write("**Dataset:** SEER")
st.write("**Task:** Multi-class classification")
st.write("**Target:** 6th Stage")

st.markdown("---")

st.markdown("### Workflow")

st.write("1. Upload trained model")
st.write("2. Upload preprocessing artifacts")
st.write("3. Upload CSV dataset")
st.write("4. Generate predictions")
st.write("5. Evaluate results")
st.write("6. Download predictions")
```

# ============================================================

# INTRODUCTION

# ============================================================

st.markdown(
'<div class="section-title">Prediction Workspace</div>',
unsafe_allow_html=True
)

st.markdown(
""" <div class="section-description">
Upload the trained XGBoost model, preprocessing artifacts,
and clinical dataset to generate breast cancer stage
predictions. </div>
""",
unsafe_allow_html=True
)

st.markdown(
""" <div class="info-box"> <strong>Required model artifacts:</strong>
xgboost_model.pkl,
target_encoder.pkl,
feature_encoders.pkl,
and feature_columns.pkl.
The uploaded dataset must be in CSV format. </div>
""",
unsafe_allow_html=True
)

# ============================================================

# FILE UPLOADS

# ============================================================

col1, col2 = st.columns(2)

with col1:

```
st.markdown("### 🧠 XGBoost Model")

model_file = st.file_uploader(
    "Upload xgboost_model.pkl",
    type=["pkl"],
    key="model"
)
```

with col2:

```
st.markdown("### 🎯 Target Encoder")

target_encoder_file = st.file_uploader(
    "Upload target_encoder.pkl",
    type=["pkl"],
    key="target"
)
```

col3, col4 = st.columns(2)

with col3:

```
st.markdown("### 🔤 Feature Encoders")

feature_encoder_file = st.file_uploader(
    "Upload feature_encoders.pkl",
    type=["pkl"],
    key="features"
)
```

with col4:

```
st.markdown("### 📋 Feature Columns")

feature_columns_file = st.file_uploader(
    "Upload feature_columns.pkl",
    type=["pkl"],
    key="columns"
)
```

st.markdown("### 📊 Clinical Dataset")

dataset_file = st.file_uploader(
"Upload the CSV dataset to be evaluated",
type=["csv"],
key="dataset"
)

# ============================================================

# CHECK FILES

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

# DATA PREVIEW

# ============================================================

if dataset_file:

```
try:

    preview_df = pd.read_csv(dataset_file)

    preview_df.columns = (
        preview_df.columns.str.strip()
    )

    st.markdown(
        '<div class="section-title">Dataset Preview</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"Dataset contains **{len(preview_df):,} records** "
        f"and **{len(preview_df.columns)} columns**."
    )

    st.dataframe(
        preview_df.head(10),
        use_container_width=True
    )

    dataset_file.seek(0)

except Exception as e:

    st.error(
        f"Unable to read the uploaded CSV: {e}"
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
# Load artifacts
# --------------------------------------------------------

model = joblib.load(
    model_file
)

target_encoder = joblib.load(
    target_encoder_file
)

feature_encoders = joblib.load(
    feature_encoder_file
)

feature_columns = joblib.load(
    feature_columns_file
)


# --------------------------------------------------------
# Load dataset
# --------------------------------------------------------

dataset_file.seek(0)

df = pd.read_csv(
    dataset_file
)

df.columns = (
    df.columns.str.strip()
)


if df.empty:

    raise ValueError(
        "The uploaded dataset is empty."
    )


# --------------------------------------------------------
# Remove completely empty unnamed columns
# --------------------------------------------------------

empty_unnamed = [

    c

    for c in df.columns

    if c.lower().startswith("unnamed")
    and df[c].isna().all()

]

if empty_unnamed:

    df = df.drop(
        columns=empty_unnamed
    )


# --------------------------------------------------------
# Target
# --------------------------------------------------------

target_column = "6th Stage"

has_target = (
    target_column in df.columns
)


# --------------------------------------------------------
# Prepare X
# --------------------------------------------------------

if has_target:

    X = df.drop(
        columns=[target_column]
    ).copy()

else:

    X = df.copy()


# --------------------------------------------------------
# Check features
# --------------------------------------------------------

missing_features = [

    col

    for col in feature_columns

    if col not in X.columns

]

if missing_features:

    raise ValueError(
        "The following required features "
        "are missing from the dataset: "
        +
        ", ".join(missing_features)
    )


# Keep the exact training feature order

X = X[
    feature_columns
].copy()


# --------------------------------------------------------
# Apply saved encoders
# --------------------------------------------------------

for col in feature_columns:

    if col not in feature_encoders:

        raise ValueError(
            f"No encoder found for feature '{col}'."
        )


    encoder = (
        feature_encoders[col]
    )


    values = (
        X[col].astype(str)
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
            f"in feature '{col}': "
            +
            ", ".join(
                map(str, examples)
            )
        )


    X[col] = encoder.transform(
        values
    )


# --------------------------------------------------------
# Prediction
# --------------------------------------------------------

predictions_encoded = (
    model.predict(
        X.values
    )
)


predictions = (
    target_encoder.inverse_transform(
        predictions_encoded.astype(int)
    )
)


# --------------------------------------------------------
# Results
# --------------------------------------------------------

result_df = df.copy()

result_df[
    "Predicted Stage"
] = predictions


# --------------------------------------------------------
# Metrics
# --------------------------------------------------------

metrics = None

if has_target:

    actual_encoded = (
        target_encoder.transform(
            df[target_column].astype(str)
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

        "confusion_matrix": cm

    }


return result_df, metrics
```

# ============================================================

# RUN PREDICTION

# ============================================================

st.markdown("---")

if all_files_uploaded:

```
st.markdown(
    '<div class="section-title">Run Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    All required files have been uploaded.
    Click the button below to process the dataset.
    """,
    unsafe_allow_html=True
)


if st.button(
    "🚀 Run XGBoost Prediction",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Loading model and generating predictions..."
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

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
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

st.markdown(
    '<div class="section-title">Prediction Results</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------------
# Metrics
# --------------------------------------------------------

if metrics:

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Records Processed",
            f"{len(result_df):,}"
        )


    with col2:

        st.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )


    with col3:

        st.metric(
            "Precision",
            f"{metrics['precision'] * 100:.2f}%"
        )


    with col4:

        st.metric(
            "F1 Score",
            f"{metrics['f1'] * 100:.2f}%"
        )


    st.markdown(
        "### Additional Evaluation"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Recall",
            f"{metrics['recall'] * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Correct Predictions",
            f"{metrics['correct']:,}"
        )


    # ----------------------------------------------------
    # Confusion Matrix
    # ----------------------------------------------------

    st.markdown(
        "### Confusion Matrix"
    )


    cm = metrics[
        "confusion_matrix"
    ]


    class_names = (
        joblib.load(
            target_encoder_file
        ).classes_
    )


    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual: {x}"
            for x in class_names
        ],
        columns=[
            f"Predicted: {x}"
            for x in class_names
        ]
    )


    st.dataframe(
        cm_df,
        use_container_width=True
    )


else:

    st.info(
        """
        The uploaded dataset does not contain the
        '6th Stage' target column. Predictions were
        generated, but accuracy, precision, recall,
        and F1 score cannot be calculated without
        known target labels.
        """
    )


# --------------------------------------------------------
# Prediction preview
# --------------------------------------------------------

st.markdown(
    "### Prediction Preview"
)


st.dataframe(
    result_df.head(50),
    use_container_width=True
)


# --------------------------------------------------------
# Download
# --------------------------------------------------------

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
    <strong>Research and Educational Use Only</strong><br>

    This application is a machine learning research
    demonstration. Predictions generated by the XGBoost
    model should not be interpreted as a medical diagnosis,
    prognosis, or treatment recommendation. Clinical
    decisions should be made by qualified healthcare
    professionals using appropriate clinical evidence.

</div>
""",
unsafe_allow_html=True
```

)

# ============================================================

# FOOTER

# ============================================================

st.markdown(
""" <div class="footer">

```
    Breast Cancer Stage Predictor
    &nbsp;•&nbsp;
    XGBoost Machine Learning Research Demonstration

</div>
""",
unsafe_allow_html=True
```

)
