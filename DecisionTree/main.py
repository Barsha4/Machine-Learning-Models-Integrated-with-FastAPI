from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from DecisionTree import DecisionTreeRegressor

# Load the trained model
with open("decision_tree_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)