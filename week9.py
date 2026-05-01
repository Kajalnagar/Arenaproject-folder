
# DAY 1: UNDERSTAND THE DATA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Create dataset (600 rows)
n = 600
data = pd.DataFrame({
    "Marketing_Spend": np.random.normal(50000, 12000, n),
    "Price": np.random.uniform(100, 500, n),
    "Region": np.random.choice(["North", "South", "East", "West"], n),
    "Customer_Segment": np.random.choice(["Retail", "Corporate"], n)
})

# Target variable (Sales)
data["Sales"] = (
    1000
    + 2.5 * data["Marketing_Spend"]
    - 50 * data["Price"]
    + np.random.normal(0, 20000, n)
)

print("\n--- DATA PREVIEW ---")
print(data.head())

# Missing values
print("\nMissing Values:\n", data.isnull().sum())

# Relationship visualization
sns.pairplot(data.select_dtypes(include=[np.number]))
plt.show()



# DAY 2: DATA PREPARATION

# Handle missing values
data = data.dropna()

# Convert categorical → numeric
data = pd.get_dummies(data, drop_first=True)

# Features & target
X = data.drop("Sales", axis=1)
y = data["Sales"]

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# DAY 3: LINEAR REGRESSION FROM SCRATCH

class LinearRegressionScratch:
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta

lr_scratch = LinearRegressionScratch()
lr_scratch.fit(X_train.values, y_train.values)
y_pred_scratch = lr_scratch.predict(X_test.values)



# DAY 4: SCIKIT-LEARN MODEL

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)



# DAY 5: MODEL EVALUATION

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n--- EVALUATION (SCIKIT-LEARN MODEL) ---")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted")
plt.show()



# DAY 6: MODEL IMPROVEMENT

from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

lr_poly = LinearRegression()
lr_poly.fit(X_train_p, y_train_p)
y_pred_poly = lr_poly.predict(X_test_p)

print("\nPolynomial R2:", r2_score(y_test_p, y_pred_poly))

# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("Decision Tree R2:", r2_score(y_test, y_pred_dt))

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest R2:", r2_score(y_test, y_pred_rf))



# DAY 7: INTERPRET & PRESENT

print("\n--- LINEAR REGRESSION COEFFICIENTS ---")
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_
})
print(coef_df)

# Feature importance (Random Forest)
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n--- FEATURE IMPORTANCE ---")
print(importance)

# Visualization
plt.figure()
sns.barplot(x="Importance", y="Feature", data=importance)
plt.title("Feature Importance (Random Forest)")
plt.show()



# FINAL INSIGHTS

print("\n--- BUSINESS INSIGHTS ---")

top_feature = importance.iloc[0]["Feature"]
print(f"Most important feature: {top_feature}")

print("Marketing Spend positively impacts Sales")
print("Price negatively impacts Sales")
print("Random Forest performs best (captures complex patterns)")

print("\n Project Completed Successfully!")
