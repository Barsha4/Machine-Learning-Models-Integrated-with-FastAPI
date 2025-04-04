from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
from LinearReg import LinearRegression
from fastapi.middleware.cors import CORSMiddleware

# Load saved model
with open("linear_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
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
    features: list[float]

# Define prediction endpoint
@app.post("/predict/")
def predict(data: InputData):
    X_new = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X_new)
    return {"prediction": prediction.tolist()}
