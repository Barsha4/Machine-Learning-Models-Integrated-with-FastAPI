from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from NaiveBayes import GaussianNB

# Load the trained Naive Bayes model
with open("naive_bayes_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict(data: InputData):
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8011)
