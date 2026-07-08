# Read CSV File
import pandas as pd

# Load Dataset
df = pd.read_csv("car data.csv")

# Display First 10 Records
print("First 10 Records")
print(df.head(10))

# Check Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# Display Column Names
print("\nColumn Names:")
print(df.columns)

# Display Data Types
print("\nDataset Information:")
df.info()

# Display Statistical Summary
print("\nStatistical Summary:")
print(df.describe())