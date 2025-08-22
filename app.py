import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

joblib.dump(clf, "breastcancer_model.joblib")

app = FastAPI()

model = joblib.load("breastcancer_model.joblib")

class BreastCancerUser(BaseModel):
    features: list[float]  

@app.get("/")
def index():
    return {"message": "Breast Cancer Prediction API"}


@app.post("/predict")
def predict_cancer(data: BreastCancerUser):
    prediction = model.predict([data.features])
    result = "Malignant" if prediction[0] == 0 else "Benign"

    return {
        "prediction": int(prediction[0]),  # 0 or 1
        "result": result
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
