
import pandas as pd
import numpy as np

# Column names
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Load dataset
df = pd.read_csv(
    'data/heart.csv',
    header=None,
    names=columns,
    na_values='?'
)

print("Original Dataset Shape:", df.shape)

# 1. Remove duplicate records
duplicates = df.duplicated().sum()
print("\nDuplicate Records:", duplicates)

df = df.drop_duplicates()

# 2. Handle missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

for column in df.columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna(
            df[column].median()
        )

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 3. Handle outliers using IQR
feature_columns = columns[:-1]

for column in feature_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[column] = df[column].clip(
        lower=lower,
        upper=upper
    )

# 4. Convert target into binary classification
# 0 = No Heart Disease
# 1, 2, 3, 4 = Heart Disease
df['target'] = (df['target'] > 0).astype(int)

# Save cleaned dataset
df.to_csv('data/heart_cleaned.csv', index=False)

print("\nPreprocessing Completed Successfully!")

print("\nFinal Dataset Shape:", df.shape)

print("\nFinal Dataset Preview:")
print(df.head())

print("\nTarget Distribution:")
print(df['target'].value_counts())

print("\nCleaned dataset saved as data/heart_cleaned.csv")
