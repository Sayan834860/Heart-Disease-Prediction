
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('data/heart_cleaned.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN Model
model = KNeighborsClassifier(n_neighbors=5)

# Train model
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("KNN MODEL RESULTS")
print("-----------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Find similar patients
print("\nSimilar Patient Analysis:")

for i in range(min(5, len(X_test))):
    distances, indices = model.kneighbors(
        [X_test[i]], n_neighbors=5
    )

    print(f"\nPatient {i+1}:")
    print("Actual Risk:", "High" if y_test.iloc[i] == 1 else "Low")
    print("Similar Patients:", indices[0] + 1)

print("\nKNN Model Completed Successfully!")
