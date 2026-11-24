
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv('data/heart_cleaned.csv')

# Create output directory
os.makedirs('outputs', exist_ok=True)

# 1. Statistical Analysis
print("STATISTICAL ANALYSIS")
print(df.describe())

# 2. Heart Disease Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='target')
plt.title('Heart Disease Distribution')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Number of Patients')
plt.savefig('outputs/heart_disease_distribution.png')
plt.show()

# 3. Age Distribution
plt.figure(figsize=(7, 5))
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Patient Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('outputs/age_distribution.png')
plt.show()

# 4. Cholesterol Analysis
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x='target', y='chol')
plt.title('Cholesterol vs Heart Disease')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Cholesterol')
plt.savefig('outputs/cholesterol_analysis.png')
plt.show()

# 5. Correlation Analysis
plt.figure(figsize=(12, 8))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png')
plt.show()

print("\nData Analysis Completed Successfully!")
print("All graphs saved in outputs folder.")
