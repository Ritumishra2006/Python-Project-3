import pandas as pd
import matplotlib.pyplot as plt

# Read CSV File
df = pd.read_csv("car data.csv")

# Check Missing Values
print("Missing Values:")
print(df.isnull().sum())

# Find Duplicate Records
print("\nDuplicate Records:", df.duplicated().sum())

# Generate Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Histogram of Selling Price
plt.figure(figsize=(8,5))
plt.hist(df["Selling_Price"], bins=15, edgecolor="black")
plt.title("Histogram of Selling Price")
plt.xlabel("Selling Price")
plt.ylabel("Frequency")
plt.show()

# Boxplots
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

for column in numeric_columns:
    plt.figure(figsize=(6,4))
    plt.boxplot(df[column])
    plt.title("Boxplot of " + column)
    plt.ylabel(column)
    plt.show()

# Scatter Plots
plt.figure(figsize=(6,4))
plt.scatter(df["Present_Price"], df["Selling_Price"])
plt.title("Present Price vs Selling Price")
plt.xlabel("Present Price")
plt.ylabel("Selling Price")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["Kms_Driven"], df["Selling_Price"])
plt.title("Kms Driven vs Selling Price")
plt.xlabel("Kms Driven")
plt.ylabel("Selling Price")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["Year"], df["Selling_Price"])
plt.title("Year vs Selling Price")
plt.xlabel("Year")
plt.ylabel("Selling Price")
plt.show()

# Correlation Heatmap
correlation = df[numeric_columns].corr()

plt.figure(figsize=(8,6))
plt.imshow(correlation, cmap="coolwarm")
plt.colorbar()

plt.xticks(range(len(numeric_columns)), numeric_columns, rotation=90)
plt.yticks(range(len(numeric_columns)), numeric_columns)

plt.title("Correlation Heatmap")
plt.show()