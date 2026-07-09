import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv("car data.csv")

# One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"],
    dtype=int
)

# Features (Independent Variables)
X = df.drop("Selling_Price", axis=1)

# Target (Dependent Variable)
y = df["Selling_Price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=2
)

# Train Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=100,
    random_state=2
)

model.fit(X_train, y_train)

print("Random Forest Model Trained Successfully!")