import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Read the dataset
df = pd.read_csv("car data.csv")

# Feature Engineering
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
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest Regression Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict on Test Data
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display Results
print("Model Evaluation")

print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)

# Interpretation
print("\nInterpretation:")
print("1. MAE: Lower MAE indicates better prediction accuracy.")
print("2. MSE: Lower MSE means fewer large prediction errors.")
print("3. RMSE: Lower RMSE indicates the model predicts selling prices more accurately.")
print("4. R² Score: Higher R² (closer to 1) means the model explains more variance in the selling prices.")






