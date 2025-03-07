from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model and feature names
with open("house_price_model.pkl", "rb") as file:
    model, columns = pickle.load(file)  # Ensure both model and column names are loaded

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("🔹 Received Form Data:", request.form)  # Debugging log

        # Extract input data from form
        input_data = {
            'Neighborhood': request.form.get('Neighborhood', ''),  # Match HTML form
            'OverallQual': int(request.form.get('OverallQual', 0)),
            'GrLivArea': int(request.form.get('GrLivArea', 0)),
            'GarageCars': int(request.form.get('GarageCars', 0)),
            'TotalBsmtSF': int(request.form.get('TotalBsmtSF', 0)),
            'YearBuilt': int(request.form.get('YearBuilt', 0)),
            'FullBath': int(request.form.get('FullBath', 0))
        }

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # One-Hot Encode input data
        input_df = pd.get_dummies(input_df)

        # Ensure input has the same features as training data
        missing_cols = {col: 0 for col in columns if col not in input_df.columns}
        missing_df = pd.DataFrame([missing_cols])

        # Concatenate to avoid performance issues
        input_df = pd.concat([input_df, missing_df], axis=1)

        # Reorder columns to match training order
        input_df = input_df[columns]

        # Make prediction
        prediction = model.predict(input_df)[0]

        return render_template('index.html', result=f'Predicted House Price: ${prediction:,.2f}')

    except KeyError as e:
        print(f"❌ Missing form key: {e}")  # Log the error
        return f"Missing form key: {e}", 400

    except Exception as e:
        print(f"❌ Error: {e}")  # Log any unexpected error
        return f"An error occurred: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)
