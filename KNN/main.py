from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from knn import KNeighborsClassifier
from fastapi.middleware.cors import CORSMiddleware

# Load the model
with open("knn_model.pkl", "rb") as f:
    knn, scaler = pickle.load(f)

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class InputData(BaseModel):
    features: list

@app.post("/predict")
async def predict(data: InputData):
    try:
        # Transform input data
        input_array = np.array(data.features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = knn.predict(input_scaled)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def home():
    return {"message": "KNN FastAPI Backend Running"}
