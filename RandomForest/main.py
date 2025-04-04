from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier  
from random_forest import RandomForestClassifier
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load("random_forest_model.pkl")  
@app.post("/predict")
async def predict(features: dict):
    try:
        input_features = np.array(features["features"]).reshape(1, -1)

        if input_features.shape[1] != 3:
            return {"error": "Expected 3 features: Weight, Waist, Pulse"}
        
        prediction = model.predict(input_features).tolist()
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}
