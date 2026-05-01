
# 1. IMPORT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

sns.set(style="whitegrid")


# 2. DATA GENERATION (500+ rows)

np.random.seed(42)
n = 600

data = pd.DataFrame({
    "Marketing_Spend": np.random.normal(50000, 12000, n),
    "Price": np.random.uniform(100, 500, n),
    "Region": np.random.choice(["North", "South", "East", "West"], n),
    "Customer_Segment": np.random.choice(["Retail", "Corporate"], n)
})

# Create realistic Sales relationship
data["Sales"] = (
    1000
    + 2.5 * data["Marketing_Spend"]
    - 50 * data["Price"]
    + np.random.normal(0, 20000, n)
)

print("\nSample Data:\n", data.head())



# 3. DATA CLEANING

print("\nMissing Values:\n", data.isnull().sum())
data.dropna(inplace=True)



# 4. DESCRIPTIVE STATISTICS

print("\n--- DESCRIPTIVE STATISTICS ---")
print(data.describe())

print("\nMean:\n", data.mean(numeric_only=True))
print("\nMedian:\n", data.median(numeric_only=True))
print("\nMode:\n", data.mode().iloc[0])
print("\nStandard Deviation:\n", data.std(numeric_only=True))



# 5. DISTRIBUTION ANALYSIS

plt.figure()
sns.histplot(data["Sales"], kde=True)
plt.title("Sales Distribution")
plt.show()

# Shapiro Test
stat, p = stats.shapiro(data["Sales"])
print("\nShapiro Test p-value:", p)

if p > 0.05:
    print("Data is likely normally distributed")
else:
    print("Data is NOT normally distributed")



# 6. CORRELATION & COVARIANCE (FIXED)

numeric_data = data.select_dtypes(include=[np.number])

corr = numeric_data.corr()
cov = numeric_data.cov()

print("\n--- CORRELATION ---")
print(corr)

print("\n--- COVARIANCE ---")
print(cov)

plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()



# 7. HYPOTHESIS TESTING (3 TESTS)


# 1. One-sample t-test
t_stat1, p_val1 = stats.ttest_1samp(data["Sales"], 150000)
print("\nOne-sample t-test p-value:", p_val1)

# 2. Independent t-test (Region comparison)
north = data[data["Region"] == "North"]["Sales"]
south = data[data["Region"] == "South"]["Sales"]

t_stat2, p_val2 = stats.ttest_ind(north, south)
print("Independent t-test (North vs South) p-value:", p_val2)

# 3. ANOVA
anova = stats.f_oneway(
    data[data["Region"] == "North"]["Sales"],
    data[data["Region"] == "South"]["Sales"],
    data[data["Region"] == "East"]["Sales"],
    data[data["Region"] == "West"]["Sales"]
)

print("ANOVA p-value:", anova.pvalue)



# 8. CONFIDENCE INTERVAL

mean = data["Sales"].mean()
sem = stats.sem(data["Sales"])

ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)

print("\n95% Confidence Interval:", ci)



# 9. REGRESSION ANALYSIS

model = smf.ols("Sales ~ Marketing_Spend + Price", data=data).fit()

print("\n--- REGRESSION SUMMARY ---")
print(model.summary())



# 10. ADDITIONAL VISUALIZATIONS (5+ total)


# Scatter plot
plt.figure()
sns.scatterplot(x="Marketing_Spend", y="Sales", data=data)
plt.title("Marketing Spend vs Sales")
plt.show()

# Boxplot
plt.figure()
sns.boxplot(x="Region", y="Sales", data=data)
plt.title("Sales by Region")
plt.show()

# Countplot
plt.figure()
sns.countplot(x="Customer_Segment", data=data)
plt.title("Customer Segment Distribution")
plt.show()



# 11. BUSINESS INSIGHTS

print("\n--- BUSINESS INSIGHTS ---")

marketing_corr = corr.loc["Sales", "Marketing_Spend"]
print(f"Correlation (Marketing vs Sales): {marketing_corr:.2f}")

if marketing_corr > 0.7:
    print("Strong positive relationship: Marketing increases sales")

if p_val2 < 0.05:
    print("Significant difference between regions")
else:
    print("No significant regional difference")

print("\nRegression Coefficients:")
print(model.params)

print(f"\nImpact: 1 unit increase in Marketing → {model.params['Marketing_Spend']:.2f} increase in Sales")



# 12. SAVE OUTPUT

data.to_csv("final_business_analysis.csv", index=False)

print("\n Project Completed Successfully!")
