from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

MODEL_PATHS = [
    "/mnt/data/best_model.pkl",
    "/mnt/data/loan_risk_model.pkl",
    "/mnt/data/loan_risk_model.joblib",
    "/mnt/data/model.pkl",
]

model = None
model_path = None

def load_model():
    global model, model_path
    for p in MODEL_PATHS:
        if os.path.exists(p):
            model = joblib.load(p)
            model_path = p
            print("Loaded model from", p)
            return
    print("No model found in default paths. Use POST /upload_model to upload one.")

load_model()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "model_loaded": model is not None, "model_path": model_path})

@app.route("/upload_model", methods=["POST"])
def upload_model():
    if 'model' not in request.files:
        return jsonify({"error":"No file part, send with key 'model'"}), 400
    file = request.files['model']
    save_path = "/mnt/data/uploaded_model.pkl"
    file.save(save_path)
    try:
        loaded = joblib.load(save_path)
        global model, model_path
        model = loaded
        model_path = save_path
        return jsonify({"message":"Model uploaded and loaded", "path": save_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error":"No model loaded. Upload one first."}), 400
    data = request.get_json()
    if not data:
        return jsonify({"error":"Expected JSON body"}), 400
    try:
        X = pd.DataFrame([data])
        for col in X.columns:
            try:
                X[col] = pd.to_numeric(X[col])
            except Exception:
                pass
        preds = model.predict(X)
        resp = {"prediction": int(preds[0]) if (hasattr(preds[0],"item") or isinstance(preds[0], (np.integer,))) else preds[0]}
        if hasattr(model, "predict_proba"):
            resp["probability"] = float(model.predict_proba(X)[:,1][0])
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if model is None:
        return jsonify({"error":"No model loaded. Upload one first."}), 400
    if 'file' not in request.files:
        return jsonify({"error":"Please upload a CSV file using form key 'file'"}), 400
    file = request.files['file']
    df = pd.read_csv(file)
    try:
        preds = model.predict(df)
        out = df.copy()
        out['prediction'] = preds
        if hasattr(model, "predict_proba"):
            out['probability'] = model.predict_proba(df)[:,1]
        out_path = "/mnt/data/batch_predictions_flask.csv"
        out.to_csv(out_path, index=False)
        return send_file(out_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)