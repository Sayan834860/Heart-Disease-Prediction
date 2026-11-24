
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load cleaned dataset
df = pd.read_csv('data/heart_cleaned.csv')

# Select patient medical features
X = df.drop('target', axis=1)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create K-Means Model
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train model
df['cluster'] = model.fit_predict(X_scaled)

# Analyze clusters using average target value
cluster_risk = df.groupby('cluster')['target'].mean()

# Sort clusters according to average risk
sorted_clusters = cluster_risk.sort_values()

# Assign risk categories
risk_categories = {
    sorted_clusters.index[0]: 'Low Risk',
    sorted_clusters.index[1]: 'Medium Risk',
    sorted_clusters.index[2]: 'High Risk'
}

df['risk_category'] = df['cluster'].map(risk_categories)

# Display results
print("K-MEANS CLUSTERING RESULTS")
print("--------------------------")

print("\nPatient Risk Distribution:")
print(df['risk_category'].value_counts())

print("\nCluster Risk Analysis:")
print(cluster_risk)

print("\nSample Patient Risk Categories:")
print(df[['age', 'sex', 'chol', 'risk_category']].head(10))

print("\nK-Means Clustering Completed Successfully!")
