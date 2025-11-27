
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Loan Risk Scoring App", layout="wide")

st.title("Loan Risk Scoring — Demo App")
st.markdown(
    """
This app loads a saved model pipeline and lets you score single records or upload a CSV for batch scoring.
It expects a scikit-learn Pipeline saved with `joblib.dump(...)` where the pipeline contains a preprocessing step (ColumnTransformer) and a classifier.
"""
)

# Attempt to find a model
candidate_paths = [
    "/mnt/data/best_model.pkl",
    "/mnt/data/loan_risk_model.pkl",
    "/mnt/data/loan_risk_model.joblib",
    "/mnt/data/model.pkl"
]
model_path = None
for p in candidate_paths:
    if os.path.exists(p):
        model_path = p
        break

if model_path is None:
    st.warning("No model found in /mnt/data. Please upload a pipeline .pkl file or place your model in /mnt/data as best_model.pkl or loan_risk_model.pkl.")
    uploaded_model = st.file_uploader("Upload pipeline (.pkl or .joblib)", type=["pkl","joblib"])
    if uploaded_model is not None:
        with open("/mnt/data/uploaded_model.pkl","wb") as f:
            f.write(uploaded_model.getbuffer())
        model_path = "/mnt/data/uploaded_model.pkl"

if model_path:
    st.success(f"Using model: {model_path}")
    model = joblib.load(model_path)

    # Try to find the preprocessor and original feature names
    pre = None
    orig_features = None
    # Common pipeline step names
    for name in ['preprocessor','prep','pre','processor','preproc']:
        if hasattr(model, 'named_steps') and name in model.named_steps:
            pre = model.named_steps[name]
            break
    # If pipeline stored directly as preprocessor then classifier inside, try to inspect
    if pre is None and hasattr(model, 'steps'):
        try:
            pre = dict(model.steps).get('preprocessor') or dict(model.steps).get('prep') or list(model.steps)[0][1]
        except Exception:
            pre = None

    # Derive original feature columns from the model if possible (fallback: user input)
    if pre is not None and hasattr(pre, 'transformers_'):
        orig_features = []
        for name, trans, cols in pre.transformers_:
            if isinstance(cols, (list, tuple, np.ndarray)):
                try:
                    orig_features.extend(list(cols))
                except Exception:
                    pass
    else:
        if hasattr(model, 'feature_names_in_'):
            try:
                orig_features = list(model.feature_names_in_)
            except Exception:
                orig_features = None

    st.sidebar.header("Scoring mode")
    mode = st.sidebar.radio("Choose mode", ["Single input", "Batch upload (CSV)"])

    if mode == "Single input":
        st.sidebar.markdown("Enter values for features below. If the app can't detect features from the model, you can paste a CSV with header or upload one.")
        if orig_features:
            st.subheader("Single record input")
            values = {}
            cols = orig_features
            for c in cols:
                v = st.text_input(f"{c}", value="")
                values[c] = v
            if st.button("Score single record"):
                X = pd.DataFrame([values])
                for col in X.columns:
                    try:
                        X[col] = pd.to_numeric(X[col])
                    except Exception:
                        pass
                try:
                    preds = model.predict(X)
                    probs = model.predict_proba(X)[:,1] if hasattr(model, "predict_proba") else None
                    st.write("Prediction:", preds[0])
                    if probs is not None:
                        st.write("Probability (positive class):", float(probs[0]))
                    st.write("Input preview:")
                    st.table(X.T)
                except Exception as e:
                    st.error("Model failed to score input. Error: " + str(e))
        else:
            st.info("Model did not expose original feature names. Upload a small CSV (one row) with column headers matching the model's training data.")
            uploaded = st.file_uploader("Upload single-row CSV", type=["csv"])
            if uploaded is not None:
                X = pd.read_csv(uploaded)
                if X.shape[0] > 1:
                    st.warning("Please upload a CSV with a single row for single-record mode. For batch scoring, choose 'Batch upload'.")
                else:
                    if st.button("Score uploaded record"):
                        try:
                            preds = model.predict(X)
                            probs = model.predict_proba(X)[:,1] if hasattr(model, "predict_proba") else None
                            st.write("Prediction:", preds[0])
                            if probs is not None:
                                st.write("Probability (positive class):", float(probs[0]))
                            st.write("Input preview:")
                            st.table(X.T)
                        except Exception as e:
                            st.error("Model failed to score input. Error: " + str(e))

    else:
        st.subheader("Batch scoring — upload CSV with same columns used at training time")
        uploaded = st.file_uploader("Upload CSV for batch scoring", type=["csv"])
        if uploaded is not None:
            X_batch = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(X_batch.head())
            if st.button("Score batch and download results"):
                try:
                    preds = model.predict(X_batch)
                    probs = model.predict_proba(X_batch)[:,1] if hasattr(model, "predict_proba") else None
                    out = X_batch.copy()
                    out['prediction'] = preds
                    if probs is not None:
                        out['probability'] = probs
                    out_path = "/mnt/data/batch_scored_results.csv"
                    out.to_csv(out_path, index=False)
                    st.success(f"Saved scored results to {out_path}")
                    st.write(out.head())
                    with open(out_path, "rb") as f:
                        st.download_button("Download scored CSV", data=f, file_name="batch_scored_results.csv")
                except Exception as e:
                    st.error("Batch scoring failed. Error: " + str(e))

    st.sidebar.markdown("---")
    st.sidebar.write("Model file used:")
    st.sidebar.code(model_path)

else:
    st.info("Please provide a model file to enable scoring.")
