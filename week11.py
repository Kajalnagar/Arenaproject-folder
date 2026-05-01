
# IMPORT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

np.random.seed(42)
sns.set(style="whitegrid")


# GENERATE CUSTOMER DATA

n = 600

data = pd.DataFrame({
    "Age": np.random.randint(18, 65, n),
    "Income": np.random.normal(50000, 15000, n),
    "Spending_Score": np.random.randint(1, 100, n),
    "Tenure": np.random.randint(1, 10, n),
    "Churn": np.random.choice([0, 1], n)
})

print("\nSample Data:\n", data.head())

# Scale data for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop("Churn", axis=1))



# DAY 1: K-MEANS + ELBOW METHOD

inertia = []

for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure()
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# Apply KMeans (choose k=4)
kmeans = KMeans(n_clusters=4, random_state=42)
data["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

print("\nKMeans Silhouette Score:", silhouette_score(X_scaled, data["Cluster_KMeans"]))



# DAY 2: HIERARCHICAL + DBSCAN


# Hierarchical clustering
linked = linkage(X_scaled, method='ward')

plt.figure()
dendrogram(linked)
plt.title("Dendrogram")
plt.show()

data["Cluster_Hierarchical"] = fcluster(linked, 4, criterion='maxclust')

# DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5)
data["Cluster_DBSCAN"] = dbscan.fit_predict(X_scaled)

print("\nDBSCAN clusters:", data["Cluster_DBSCAN"].unique())



# DAY 3: SEGMENT ANALYSIS

segment_profile = data.groupby("Cluster_KMeans").mean()
print("\n--- SEGMENT PROFILE ---")
print(segment_profile)

# Name segments
segment_names = {
    0: "High Value",
    1: "Budget",
    2: "Mid-Level",
    3: "Premium"
}

data["Segment_Name"] = data["Cluster_KMeans"].map(segment_names)

print("\nSegment Distribution:\n", data["Segment_Name"].value_counts())



# DAY 4: SEGMENT-WISE MODELS

results = {}

for segment in data["Cluster_KMeans"].unique():
    segment_data = data[data["Cluster_KMeans"] == segment]

    X = segment_data.drop(["Churn", "Cluster_KMeans", "Cluster_Hierarchical", "Cluster_DBSCAN", "Segment_Name"], axis=1)
    y = segment_data["Churn"]

    if len(segment_data) < 20:
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[segment] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }



# DAY 5: MODEL EVALUATION

print("\n--- SEGMENT MODEL PERFORMANCE ---")
for seg, metrics in results.items():
    print(f"\nSegment {seg}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")



# DAY 6: HYPERPARAMETER TUNING

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10]
}

# Use full dataset for tuning
X_full = data.drop(["Churn", "Cluster_KMeans", "Cluster_Hierarchical", "Cluster_DBSCAN", "Segment_Name"], axis=1)
y_full = data["Churn"]

rf = RandomForestClassifier(random_state=42)

grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
grid.fit(X_full, y_full)

print("\nBest Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)



# DAY 7: BUSINESS 

print("\n--- BUSINESS  ---")

print("1. High Value Segment → Focus retention campaigns")
print("2. Budget Segment → Offer discounts & promotions")
print("3. Premium Segment → Provide loyalty rewards")
print("4. Mid-Level Segment → Upsell opportunities")

print("\nEstimated Impact:")
print("- Reduce churn by 10% in high-value customers")
print("- Increase revenue via targeted marketing")

print("\n PROJECT COMPLETED SUCCESSFULLY!")
