
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cleaned dataset
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

# Create Linear Regression Model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("LINEAR REGRESSION RESULTS")
print("------------------------")
print("Mean Absolute Error:", round(mae, 4))
print("Mean Squared Error:", round(mse, 4))
print("R2 Score:", round(r2, 4))

# Display sample risk scores
print("\nSample Patient Risk Scores:")

for i in range(min(5, len(predictions))):
    risk_score = max(0, min(100, predictions[i] * 100))
    print(f"Patient {i+1}: {risk_score:.2f}%")

print("\nLinear Regression Completed Successfully!")
