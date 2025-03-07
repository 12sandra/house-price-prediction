import pickle
import pandas as pd

# Load the saved model and feature names
with open("house_price_model.pkl", "rb") as file:
    model, columns = pickle.load(file)

# Example input (replace this with actual input from your web form)
input_data = {
    'Neighborhood': 'CollgCr',  # Example categorical input
    'OverallQual': 7,           # Example numerical input
    'GrLivArea': 1710,
    'GarageCars': 2,
    'TotalBsmtSF': 856,
    'YearBuilt': 2003,
    'FullBath': 2
}

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# One-Hot Encode input data
input_df = pd.get_dummies(input_df)

# Ensure input has the same features as training data
input_df = input_df.reindex(columns=columns, fill_value=0)  # Efficient way to add missing columns

# Make a prediction
prediction = model.predict(input_df)
print("Predicted House Price:", prediction[0])
