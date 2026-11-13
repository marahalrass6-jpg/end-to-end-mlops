from __future__ import annotations
import json, os
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parents[1]; MODEL_DIR=Path(os.getenv('MODEL_DIR', ROOT/'artifacts'))
app=FastAPI(title='End-to-End MLOps Prediction API', version='1.0.0')
model=None; metadata={}

class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=30, max_length=30)

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    model_version: str

@app.on_event('startup')
def load_artifact():
    global model, metadata
    model_path=MODEL_DIR/'model.joblib'; metadata_path=MODEL_DIR/'metadata.json'
    if model_path.exists() and metadata_path.exists():
        model=joblib.load(model_path); metadata=json.loads(metadata_path.read_text())

@app.get('/health')
def health(): return {'status':'ok'}

@app.get('/ready')
def ready():
    if model is None: raise HTTPException(status_code=503, detail='Model artifact is not loaded')
    return {'status':'ready','model_version':metadata.get('version')}

@app.get('/metadata')
def get_metadata():
    if not metadata: raise HTTPException(status_code=503, detail='Metadata unavailable')
    return metadata

@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None: raise HTTPException(status_code=503, detail='Model artifact is not loaded')
    frame=pd.DataFrame([request.features], columns=metadata['features'])
    probability=float(model.predict_proba(frame)[0,1]); prediction=int(probability>=.5)
    return {'prediction':prediction,'label':'positive' if prediction else 'negative','probability':probability,'model_version':metadata['version']}
