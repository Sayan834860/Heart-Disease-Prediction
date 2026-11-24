
import joblib
import pandas as pd

# Load trained model
model = joblib.load('models/heart_disease_model.pkl')

# Patient information
patient = pd.DataFrame([{
    'age': 55,
    'sex': 1,
    'cp': 2,
    'trestbps': 130,
    'chol': 250,
    'fbs': 0,
    'restecg': 1,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 1.5,
    'slope': 1,
    'ca': 0,
    'thal': 2
}])

# Prediction
prediction = model.predict(patient)[0]
probability = model.predict_proba(patient)[0]

print("HEART DISEASE PREDICTION")
print("-----------------------")

if prediction == 1:
    print("Prediction: Heart Disease Detected")
else:
    print("Prediction: No Heart Disease Detected")

print("No Disease Probability:", round(probability[0] * 100, 2), "%")
print("Heart Disease Probability:", round(probability[1] * 100, 2), "%")

print("\nPrediction Completed Successfully!")
