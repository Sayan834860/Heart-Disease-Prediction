
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load cleaned dataset
df = pd.read_csv('data/heart_cleaned.csv')

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

# 1. Logistic Regression Prediction
logistic_model = joblib.load(
    'models/heart_disease_model.pkl'
)

prediction = logistic_model.predict(patient)[0]
probability = logistic_model.predict_proba(patient)[0]

# 2. Linear Regression Risk Score
X = df.drop('target', axis=1)
y = df['target']

scaler_lr = StandardScaler()
X_scaled = scaler_lr.fit_transform(X)

linear_model = LinearRegression()
linear_model.fit(X_scaled, y)

patient_scaled = scaler_lr.transform(patient)
risk_prediction = linear_model.predict(patient_scaled)[0]

risk_score = max(0, min(100, risk_prediction * 100))

# 3. K-Means Risk Category
scaler_kmeans = StandardScaler()
X_kmeans = scaler_kmeans.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_kmeans)

df['cluster'] = clusters

cluster_risk = df.groupby('cluster')['target'].mean()
sorted_clusters = cluster_risk.sort_values()

risk_categories = {
    sorted_clusters.index[0]: 'Low Risk',
    sorted_clusters.index[1]: 'Medium Risk',
    sorted_clusters.index[2]: 'High Risk'
}

patient_kmeans = scaler_kmeans.transform(patient)
patient_cluster = kmeans.predict(patient_kmeans)[0]

risk_category = risk_categories[patient_cluster]

# 4. Display Final Results
print("\n================================")
print(" FINAL PATIENT RISK ANALYSIS")
print("================================")

print("\nPATIENT INFORMATION")
print("-------------------")
print(patient.to_string(index=False))

print("\nHEART DISEASE PREDICTION")
print("-----------------------")

if prediction == 1:
    print("Prediction: Heart Disease Detected")
else:
    print("Prediction: No Heart Disease Detected")

print(
    "No Disease Probability:",
    round(probability[0] * 100, 2), "%"
)

print(
    "Heart Disease Probability:",
    round(probability[1] * 100, 2), "%"
)

print("\nPATIENT RISK SCORE")
print("------------------")
print("Risk Score:", round(risk_score, 2), "%")

print("\nPATIENT RISK CATEGORY")
print("---------------------")
print("Risk Category:", risk_category)

print("\n================================")
print(" FINAL ANALYSIS COMPLETED")
print("================================")
