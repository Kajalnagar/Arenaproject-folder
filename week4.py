# Import libraries
import pandas as pd              # For data analysis
import matplotlib.pyplot as plt 

# Load Dataset


df = pd.read_csv("sales_data.csv")

print("Dataset Preview:")
print(df.head())

# Data Cleaning

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Data Analysis
# Create Revenue column
df["Revenue"] = df["Quantity"] * df["Price"]

# Total revenue
total_revenue = df["Revenue"].sum()

# Best selling product
best_product = df.groupby("Product")["Quantity"].sum()

print("\nTotal Revenue:", total_revenue)

print("\nSales by Product:")
print(best_product)


# Visualization 1 (Bar Chart)


plt.figure()

best_product.plot(kind="bar")

plt.title("Sales by Product")
plt.xlabel("Product")
plt.ylabel("Total Quantity Sold")

plt.show()


# Visualization 2 (Line Chart)


monthly_sales = df.groupby("Month")["Revenue"].sum()

plt.figure()

monthly_sales.plot(kind="line", marker="o")

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")

plt.show()


# Visualization 3 (Pie Chart)


city_sales = df.groupby("City")["Revenue"].sum()

plt.figure()

city_sales.plot(kind="pie", autopct="%1.1f%%")

plt.title("Revenue Distribution by City")

plt.ylabel("")

plt.show()


# Final Report


print("\n===== SALES REPORT =====")

print("Total Revenue:", total_revenue)

print("Best Selling Product:", best_product.idxmax())

print("Highest Revenue Month:", monthly_sales.idxmax())
