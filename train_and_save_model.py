import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("train.csv")  # Make sure the path is correct

# Define target column
target_column = 'SalePrice'

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# One-Hot Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Save the column names for future encoding
columns = X.columns.tolist()

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Save the model and column names
with open("house_price_model.pkl", "wb") as file:
    pickle.dump((model, columns), file)

print("✅ Model trained and saved successfully!")
