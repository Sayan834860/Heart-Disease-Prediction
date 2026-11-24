
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load cleaned dataset
df = pd.read_csv('data/heart_cleaned.csv')

# Separate input features and target
X = df.drop('target', axis=1)
y = df['target']

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display dataset information
print("Dataset Preparation Completed!")

print("\nTotal Records:", len(df))
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

print("\nTraining Data Shape:", X_train_scaled.shape)
print("Testing Data Shape:", X_test_scaled.shape)

print("\nFeatures:")
print(list(X.columns))

print("\nTarget Distribution:")
print(y.value_counts())
