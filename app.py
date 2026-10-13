import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="KFC Sales Prediction", page_icon="🍗", layout="wide")

MODEL_PATH = Path(__file__).parent / "kfc_decision_tree_model.pkl"

st.title("🍗 KFC Sales Prediction")
st.write(
    "Upload a CSV file containing the same input features used to train your "
    "Decision Tree model. The app will generate predictions for each row."
)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Add `kfc_decision_tree_model.pkl` to the same "
        "GitHub repository folder as `app.py`."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

uploaded_file = st.file_uploader("Upload input CSV", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()

    st.subheader("Preview of uploaded data")
    st.dataframe(data.head(10), use_container_width=True)
    st.write(f"Rows: {len(data)} | Columns: {len(data.columns)}")

    # The fitted preprocessing pipeline records the original feature names.
    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
    expected_columns = None
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        expected_columns = list(preprocessor.feature_names_in_)

    if expected_columns:
        missing = [col for col in expected_columns if col not in data.columns]
        extra = [col for col in data.columns if col not in expected_columns]

        if missing:
            st.error(
                "The uploaded CSV is missing required model input columns: "
                + ", ".join(missing)
            )
            st.info(
                "Upload a CSV with the same feature columns used during training. "
                "Do not include the target column that the model was trained to predict."
            )
            st.stop()

        X_input = data[expected_columns].copy()
        if extra:
            st.warning(
                "Extra columns will be ignored: " + ", ".join(extra)
            )
    else:
        st.warning(
            "The model's expected feature names could not be read automatically. "
            "The app will pass all uploaded columns to the model."
        )
        X_input = data.copy()

    if st.button("Generate predictions", type="primary"):
        try:
            predictions = model.predict(X_input)
            results = data.copy()
            results["Predicted_Value"] = predictions

            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)

            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download predictions as CSV",
                data=csv_bytes,
                file_name="kfc_sales_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info(
                "Check that the uploaded CSV has the same feature columns and "
                "compatible data types as the data used to train the model."
            )

st.divider()
st.caption("Built with Streamlit and scikit-learn.")
