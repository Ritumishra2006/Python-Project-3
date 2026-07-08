import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("car data.csv")

# Features (Independent Variables)
X = df.drop("Selling_Price", axis=1)

# Target (Dependent Variable)
y = df["Selling_Price"]

# Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Display the size of each dataset
print("Training Features Shape :", X_train.shape)
print("Testing Features Shape  :", X_test.shape)
print("Training Target Shape   :", y_train.shape)
print("Testing Target Shape    :", y_test.shape)