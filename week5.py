import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("sales_data.csv")

# Show first rows
print(df.head())

# Dataset information
print(df.info())

# Check missing values
print(df.isnull().sum())


# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# If Total_Sales not calculated, create it
df['Total_Sales'] = df['Quantity'] * df['Price']


# Customer Analysis
customer_sales = df.groupby('Customer_ID')['Total_Sales'].sum()

top_customers = customer_sales.sort_values(ascending=False).head()

print("Top Customers:")
print(top_customers)


# Regional distribution
region_distribution = df.groupby('Region')['Customer_ID'].nunique()

print("Customer Distribution by Region:")
print(region_distribution)


# Sales Pattern Analysis

# Extract month from Date
df['Month'] = df['Date'].dt.month

monthly_sales = df.groupby('Month')['Total_Sales'].sum()

print("Monthly Sales:")
print(monthly_sales)


# Best selling products
best_products = df.groupby('Product')['Quantity'].sum().sort_values(ascending=False).head()

print("Best Selling Products:")
print(best_products)


# Pivot Table
pivot_table = pd.pivot_table(df,
                             values='Total_Sales',
                             index='Region',
                             columns='Product',
                             aggfunc='sum')

print("Pivot Table:")
print(pivot_table)


# Dashboard Charts

monthly_sales.plot(kind='bar')
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.show()

region_sales = df.groupby('Region')['Total_Sales'].sum()
region_sales.plot(kind='pie', autopct='%1.1f%%')

plt.title("Sales by Region")
plt.show()
