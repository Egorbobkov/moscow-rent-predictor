from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import numpy as np
from catboost import CatBoostRegressor
from pydantic import BaseModel

app = FastAPI(title="Moscow Rent Predictor API")

model = CatBoostRegressor()
model.load_model("models/catboost_model.cbm")
feature_columns = joblib.load("models/feature_columns.pkl")

df = pd.read_csv("data/cian_train_10features.csv")


class PredictRequest(BaseModel):
    url: str


@app.post("/predict")
def predict_rent(request: PredictRequest):
    url = request.url
    row = df[df["url"] == url]
    if row.empty:
        raise HTTPException(status_code=404, detail="Объявление не найдено в датасете")

    features = row[feature_columns]

    log_pred = model.predict(features)[0]
    price = float(np.exp(log_pred))

    return {"url": url, "predicted_rent": round(price)}
