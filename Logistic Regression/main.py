from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
from LogisticReg import LogisticRegression
from fastapi.middleware.cors import CORSMiddleware

# Load trained models and scaler
with open("model.pkl", "rb") as f:
    models = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class PredictionInput(BaseModel):
    data: list
    model_type: str

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        if input_data.model_type not in models:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        model = models[input_data.model_type]
        data = np.array(input_data.data).reshape(1, -1)
        data = scaler.transform(data)
        prediction = model.predict(data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
