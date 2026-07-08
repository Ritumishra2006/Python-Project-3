import pandas as pd
df = pd.read_csv("car data.csv")

# Feature 1: Car_Age
# Car_Age = 2025 - Year
df["Car_Age"] = 2025 - df["Year"]

# Feature 2: Mileage_per_Year
# Mileage_per_Year = Kms_Driven / (Car_Age + 1)
df["Mileage_per_Year"] = df["Kms_Driven"] / (df["Car_Age"] + 1)

# Feature 3: Premium_Brand
# Premium Brands:
# Toyota, Honda, Hyundai, BMW, Audi, Mercedes
premium_brands = ["Toyota", "Honda", "Hyundai", "BMW", "Audi", "Mercedes"]

df["Premium_Brand"] = df["Car_Name"].apply(
    lambda x: 1 if any(brand in x for brand in premium_brands) else 0
)

# Feature 4: High_Mileage
# If Kms_Driven > Median
# Then 1
# Else 0
median_kms = df["Kms_Driven"].median()
df["High_Mileage"] = df["Kms_Driven"].apply(
    lambda x: 1 if x > median_kms else 0
)

# Display First 10 Records
print("First 10 Records with New Features")
print(df.head(10))