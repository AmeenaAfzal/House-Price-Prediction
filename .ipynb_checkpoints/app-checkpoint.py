from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the model and feature columns when the app starts
try:
    model = joblib.load("gradient_boost_house_price_model.pkl")
    feature_columns = joblib.load("model_features.pkl")
    print("Model and features loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {e}")
    model = None
    feature_columns = []

@app.route('/', methods=['GET'])
def home():
    # Show the main HTML page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not feature_columns:
        return render_template('index.html', prediction_text="Error: Model files not found on the server.")

    try:
        # 1. Get values from the HTML form
        overall_qual = int(request.form.get('overall_qual', 0))
        gr_liv_area = float(request.form.get('gr_liv_area', 0))
        total_bsmt_sf = float(request.form.get('total_bsmt_sf', 0))
        garage_cars = int(request.form.get('garage_cars', 0))
        year_built = int(request.form.get('year_built', 0))

        # 2. Create an empty dataframe with all features initialized to 0
        input_data = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

        # 3. Populate the dataframe with the user inputs
        if 'OverallQual' in input_data.columns: input_data.at[0, 'OverallQual'] = overall_qual
        if 'GrLivArea' in input_data.columns: input_data.at[0, 'GrLivArea'] = gr_liv_area
        if 'TotalBsmtSF' in input_data.columns: input_data.at[0, 'TotalBsmtSF'] = total_bsmt_sf
        if 'GarageCars' in input_data.columns: input_data.at[0, 'GarageCars'] = garage_cars
        if 'YearBuilt' in input_data.columns: input_data.at[0, 'YearBuilt'] = year_built

        # 4. Make the prediction
        prediction = model.predict(input_data)[0]

        # 5. Format the result and send it back to the HTML page
        result_text = f"${prediction:,.2f}"
        
        return render_template('index.html', prediction_text=f'Estimated House Price: {result_text}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    # Run the app locally on port 5000
    app.run(debug=True, port=5000)