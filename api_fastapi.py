from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import joblib
import pandas as pd
import numpy as np
import uvicorn
import os

app = FastAPI(title="Loan Risk Scoring API", version="1.0")

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
    print("No model found in default paths. API will accept model uploads via /upload_model.")

load_model()

class SinglePrediction(BaseModel):
    data: Dict[str, Any]
    return_proba: Optional[bool] = True

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_path": model_path}

@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    """
    Upload a trained pipeline file (.pkl or .joblib) and save to /mnt/data/uploaded_model.pkl
    """
    try:
        contents = await file.read()
        save_path = "/mnt/data/uploaded_model.pkl"
        with open(save_path, "wb") as f:
            f.write(contents)
        # attempt to load
        loaded = joblib.load(save_path)
        # set as current model
        global model, model_path
        model = loaded
        model_path = save_path
        return {"message": "Model uploaded and loaded", "path": save_path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to upload/load model: {e}")

@app.post("/predict")
def predict(payload: SinglePrediction):
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded. Upload a model first at /upload_model.")
    try:
        X = pd.DataFrame([payload.data])
        # attempt numeric conversion where appropriate
        for col in X.columns:
            try:
                X[col] = pd.to_numeric(X[col])
            except Exception:
                pass
        preds = model.predict(X)
        result = {"prediction": int(preds[0]) if (hasattr(preds[0], "item") or isinstance(preds[0], (np.integer,))) else preds[0]}
        if payload.return_proba:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[:,1]
                result["probability"] = float(proba[0])
            else:
                result["probability"] = None
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...), return_proba: Optional[bool] = True):
    """
    Upload a CSV and receive predictions as a CSV saved to /mnt/data/batch_predictions.csv
    """
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded. Upload a model first at /upload_model.")
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        preds = model.predict(df)
        out = df.copy()
        out["prediction"] = preds
        if return_proba and hasattr(model, "predict_proba"):
            out["probability"] = model.predict_proba(df)[:,1]
        out_path = "/mnt/data/batch_predictions.csv"
        out.to_csv(out_path, index=False)
        return {"message": "Batch predictions saved", "path": out_path, "n_rows": out.shape[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)