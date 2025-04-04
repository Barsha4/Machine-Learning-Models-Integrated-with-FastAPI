from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from svm import SVC

with open("svm_model.pkl", "rb") as model_file:
    scaler, model = pickle.load(model_file)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    features: list

@app.post("/predict/")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    X = scaler.transform(X)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X).tolist()
    return {"prediction": int(prediction), "probability": probability}

@app.get("/")
def home():
    return {"message": "SVM Model API is running with Wine Dataset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)
