import pickle as pkl
import json
import pandas as pd

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List


app = FastAPI(title='CarsAPI', description='Service for predicting car prices', version='1.0.0')

with open('../data/pipeline.pkl', 'rb') as f:
    pipeline = pkl.load(file=f)


class Item(BaseModel):
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    torque: str
    seats: float
    max_torque_rpm: str
    brand: str


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([json.loads(item.model_dump_json())])
    
    return pipeline.predict(X=df)[0]


@app.post("/predict_items")
def predict_items(csv: UploadFile):
    df = pd.read_csv(csv.file)
    df['prediction'] = pipeline.predict(X=df)
    df.to_csv('../data/predicted_data.csv', index=False)
    
    csv.file.close()

    return FileResponse('../data/predicted_data.csv')
    