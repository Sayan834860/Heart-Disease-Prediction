
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv('data/heart_cleaned.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Create model pipeline
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X, y)

# Save trained model
joblib.dump(model, 'models/heart_disease_model.pkl')

print("Model Saved Successfully!")
print("Saved Location: models/heart_disease_model.pkl")
