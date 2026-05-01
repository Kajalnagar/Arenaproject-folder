
# IMPORT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


# GENERATE SAMPLE BUSINESS DATA

np.random.seed(42)

n = 200

data = pd.DataFrame({
    "Marketing_Spend": np.random.normal(50000, 10000, n),
    "Sales": np.random.normal(200000, 40000, n),
    "Region": np.random.choice(["North", "South", "East", "West"], n)
})

# Add relationship (realistic business effect)
data["Sales"] += 2 * data["Marketing_Spend"]

print("Sample Data:\n", data.head())



# DAY 1: DESCRIPTIVE STATISTICS

print("\n--- DESCRIPTIVE STATISTICS ---")
print("Mean:\n", data.mean(numeric_only=True))
print("Median:\n", data.median(numeric_only=True))
print("Mode:\n", data.mode().iloc[0])
print("Standard Deviation:\n", data.std(numeric_only=True))



# DAY 2: DISTRIBUTION ANALYSIS

plt.figure()
sns.histplot(data["Sales"], kde=True)
plt.title("Sales Distribution")
plt.show()

# Normality Test
stat, p = stats.shapiro(data["Sales"])
print("\nShapiro Test p-value:", p)

if p > 0.05:
    print("Data is likely normally distributed")
else:
    print("Data is NOT normally distributed")



# DAY 3: CORRELATION ANALYSIS

print("\n--- CORRELATION ---")
corr = data.corr(numeric_only=True)
print(corr)

plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\nCovariance:\n", data.cov(numeric_only=True))



# DAY 4: HYPOTHESIS TESTING


# 1. One-sample t-test
t_stat, p_val = stats.ttest_1samp(data["Sales"], 250000)
print("\nOne-sample t-test p-value:", p_val)

# 2. Independent t-test (North vs South)
north = data[data["Region"] == "North"]["Sales"]
south = data[data["Region"] == "South"]["Sales"]

t_stat, p_val = stats.ttest_ind(north, south)
print("Independent t-test p-value:", p_val)

# 3. ANOVA
anova = stats.f_oneway(
    data[data["Region"] == "North"]["Sales"],
    data[data["Region"] == "South"]["Sales"],
    data[data["Region"] == "East"]["Sales"],
    data[data["Region"] == "West"]["Sales"]
)

print("ANOVA p-value:", anova.pvalue)



# DAY 5: CONFIDENCE INTERVAL

mean = data["Sales"].mean()
std_err = stats.sem(data["Sales"])

ci = stats.t.interval(
    confidence=0.95,
    df=len(data)-1,
    loc=mean,
    scale=std_err
)

print("\n95% Confidence Interval:", ci)



# DAY 6: REGRESSION ANALYSIS

model = smf.ols("Sales ~ Marketing_Spend", data=data).fit()

print("\n--- REGRESSION SUMMARY ---")
print(model.summary())



# DAY 7: BUSINESS INSIGHTS


print("\n--- BUSINESS INSIGHTS ---")

# Correlation Insight
corr_value = corr.loc["Sales", "Marketing_Spend"]
print(f"Correlation between Marketing Spend and Sales: {corr_value:.2f}")

# Interpretation
if corr_value > 0.7:
    print("Strong positive relationship: Increasing marketing likely boosts sales")

# Hypothesis interpretation
if p_val < 0.05:
    print("Statistically significant difference between groups")
else:
    print("No significant difference between groups")

print(f"Expected Sales increase per unit marketing spend: {model.params['Marketing_Spend']:.2f}")
