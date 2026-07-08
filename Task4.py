import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("car data.csv")

# Create LabelEncoder object
le = LabelEncoder()

# Apply Label Encoding
df["Car_Name"] = le.fit_transform(df["Car_Name"])
df["Fuel_Type"] = le.fit_transform(df["Fuel_Type"])
df["Seller_Type"] = le.fit_transform(df["Seller_Type"])
df["Transmission"] = le.fit_transform(df["Transmission"])

# Display first 5 rows
print(df.head())

# Save transformed dataset
df.to_csv("car_data_label_encoded.csv", index=False)

print("Label Encoding completed successfully!")


