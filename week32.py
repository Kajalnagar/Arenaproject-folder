
import pandas as pd


# Day 1: Setup & Load Data


# Read the CSV file
# read_csv() loads the dataset into a DataFrame
df = pd.read_csv("sales.csv")

# Display full dataset
print("Full Dataset:")
print(df)

# Show first 5 rows of dataset
# head() is useful for quickly checking data
print("\nFirst 5 Rows:")
print(df.head())


# Day 2: Explore Data


# Shape shows number of rows and columns
print("\nShape of dataset (rows, columns):")
print(df.shape)

# Show column names
print("\nColumn Names:")
print(df.columns)

# Show data types of each column
print("\nData Types:")
print(df.dtypes)

# Show complete summary of dataset
print("\nDataset Information:")
print(df.info())


#Day 3: Clean Data


# Check missing values in each column
print("\nMissing Values:")
print(df.isnull().sum())

# Replace missing values in Quantity column with average
df["Quantity"].fillna(df["Quantity"].mean(), inplace=True)

# Remove duplicate rows if they exist
df.drop_duplicates(inplace=True)
# Show cleaned dataset
print("\nCleaned Dataset:")
print(df)



# Day 4: Analyze Sales
# Create a new column Revenue
# Revenue = Quantity * Price
df["Revenue"] = df["Quantity"] * df["Price"]

# Show dataset with revenue column
print("\nDataset with Revenue:")
print(df)

# Calculate total revenue
total_sales = df["Revenue"].sum()

# Calculate average price
average_price = df["Price"].mean()

# Find highest price
highest_price = df["Price"].max()

# Find lowest price
lowest_price = df["Price"].min()

# Find best selling product
# groupby() groups data by product
best_product = df.groupby("Product")["Quantity"].sum().idxmax()



# Day 5: Create Final Report


print("\n==== SALES REPORT ====")

# Print total revenue
print("Total Revenue:", total_sales)

# Print statistics
print("Average Price:", average_price)
print("Highest Price:", highest_price)
print("Lowest Price:", lowest_price)

# Print best selling product
print("Best Selling Product:", best_product)

# Show statistical summary of dataset
print("\nStatistical Summary:")
print(df.describe())
