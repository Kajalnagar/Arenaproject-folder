
# DAY 1: EXPLORE & UNDERSTAND DATA

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)

# Create synthetic churn dataset (700 rows)
n = 700
data = pd.DataFrame({
    "Age": np.random.randint(18, 65, n),
    "Tenure": np.random.randint(1, 60, n),
    "Monthly_Charges": np.random.uniform(20, 120, n),
    "Total_Charges": np.random.uniform(100, 8000, n),
    "Contract_Type": np.random.choice(["Month-to-Month", "One-Year", "Two-Year"], n),
    "Payment_Method": np.random.choice(["Credit Card", "Debit Card", "UPI", "Cash"], n),
    "Churn": np.random.choice([0, 1], n)
})

print("\n--- DATA PREVIEW ---")
print(data.head())

print("\n--- DATA TYPES ---")
print(data.dtypes)

# Churn distribution
sns.countplot(x="Churn", data=data)
plt.title("Churn Distribution")
plt.show()



# DAY 2: HANDLE CATEGORICAL DATA

from sklearn.preprocessing import LabelEncoder

data_encoded = data.copy()

# Label Encoding (example: Payment Method)
le = LabelEncoder()
data_encoded["Payment_Method_Label"] = le.fit_transform(data_encoded["Payment_Method"])

# One-Hot Encoding
data_encoded = pd.get_dummies(data_encoded, columns=["Contract_Type"], drop_first=True)

# Drop original categorical columns
data_encoded = data_encoded.drop(["Payment_Method"], axis=1)

print("\n--- AFTER ENCODING ---")
print(data_encoded.head())



# DAY 3: FEATURE SCALING

from sklearn.preprocessing import MinMaxScaler, StandardScaler

num_cols = ["Age", "Tenure", "Monthly_Charges", "Total_Charges"]

# Min-Max Scaling
minmax = MinMaxScaler()
data_minmax = data_encoded.copy()
data_minmax[num_cols] = minmax.fit_transform(data_minmax[num_cols])

# Standard Scaling
standard = StandardScaler()
data_standard = data_encoded.copy()
data_standard[num_cols] = standard.fit_transform(data_standard[num_cols])

print("\n--- SCALED DATA (MinMax Sample) ---")
print(data_minmax.head())



# DAY 4: OUTLIER DETECTION & HANDLING


# IQR Method
def remove_outliers_iqr(df, cols):
    df_clean = df.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

data_iqr = remove_outliers_iqr(data_encoded, num_cols)

# Z-score Method
from scipy.stats import zscore

z_scores = np.abs(zscore(data_encoded[num_cols]))
data_z = data_encoded[(z_scores < 3).all(axis=1)]

print("\nOriginal size:", data_encoded.shape)
print("After IQR:", data_iqr.shape)
print("After Z-score:", data_z.shape)



# DAY 5: FEATURE ENGINEERING


data_fe = data_encoded.copy()

# Customer Lifetime Value (CLV)
data_fe["CLV"] = data_fe["Monthly_Charges"] * data_fe["Tenure"]

# Payment Efficiency
data_fe["Payment_Efficiency"] = data_fe["Total_Charges"] / (data_fe["Tenure"] + 1)

print("\n--- FEATURE ENGINEERING ---")
print(data_fe.head())



# DAY 6: FEATURE SELECTION


# Correlation
corr = data_fe.corr()

plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Feature Importance using Random Forest
from sklearn.ensemble import RandomForestClassifier

X = data_fe.drop("Churn", axis=1)
y = data_fe["Churn"]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n--- FEATURE IMPORTANCE ---")
print(importance)

plt.figure()
sns.barplot(x="Importance", y="Feature", data=importance)
plt.title("Feature Importance")
plt.show()



# DAY 7: BUILD COMPLETE PIPELINE

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Define columns
numeric_features = ["Age", "Tenure", "Monthly_Charges", "Total_Charges"]
categorical_features = ["Contract_Type", "Payment_Method"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

# Final pipeline with model
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train-test split
from sklearn.model_selection import train_test_split

X_raw = data.drop("Churn", axis=1)
y_raw = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# Train pipeline
pipeline.fit(X_train, y_train)

# Test predictions
y_pred = pipeline.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
print("\nPipeline Accuracy:", accuracy_score(y_test, y_pred))


print("\nCOMPLETE PREPROCESSING PIPELINE EXECUTED SUCCESSFULLY!")
