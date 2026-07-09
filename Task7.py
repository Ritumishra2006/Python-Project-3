import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
df = pd.read_csv("car data.csv")

# Create New Feature
df["Car_Age"] = 2025 - df["Year"]

# One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"],
    dtype=int
)

# Features and Target
X = df.drop(["Selling_Price", "Year"], axis=1)
y = df["Selling_Price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict Selling Prices
y_pred = model.predict(X_test)

# Compare Actual vs Predicted Prices
comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("Actual vs Predicted Prices:")
print(comparison)

# Save Comparison
comparison.to_csv("actual_vs_predicted.csv", index=False)
print("\nComparison saved as 'actual_vs_predicted.csv'")