from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# 1. Load Model, Features, AND the Scaler
try:
    model = joblib.load("gradient_boost_house_price_model.pkl")
    feature_columns = joblib.load("model_features.pkl")
    scaler = joblib.load("scaler.pkl")  # <--- NEW: Load the scaler
except Exception as e:
    model = None
    feature_columns = []
    scaler = None
    print(f"Error loading files: {e}")

# These are the exact columns your notebook applied the scaler to
num_cols = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 
            '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return render_template('index.html', prediction_text="Error: Model or Scaler not loaded.")

    try:
        # Get inputs from the form
        overall_qual = int(request.form.get('overall_qual'))
        gr_liv_area = float(request.form.get('gr_liv_area'))
        total_bsmt_sf = float(request.form.get('total_bsmt_sf'))
        garage_cars = int(request.form.get('garage_cars'))
        year_built = int(request.form.get('year_built'))

        # Create a dataframe with 0s
        input_data = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

        # Populate the dataframe with raw values
        if 'OverallQual' in input_data.columns: input_data.at[0, 'OverallQual'] = overall_qual
        if 'GrLivArea' in input_data.columns: input_data.at[0, 'GrLivArea'] = gr_liv_area
        if 'TotalBsmtSF' in input_data.columns: input_data.at[0, 'TotalBsmtSF'] = total_bsmt_sf
        if 'GarageCars' in input_data.columns: input_data.at[0, 'GarageCars'] = garage_cars
        if 'YearBuilt' in input_data.columns: input_data.at[0, 'YearBuilt'] = year_built

        # ---> THE MAGIC FIX: Scale the numeric columns before predicting! <---
        # Only scale columns that actually exist in the user's feature_columns list
        valid_num_cols = [col for col in num_cols if col in input_data.columns]
        if valid_num_cols:
            input_data[valid_num_cols] = scaler.transform(input_data[valid_num_cols])

        # Make prediction
        prediction = model.predict(input_data)[0]
        result_text = f"${prediction:,.2f}"

        return render_template('index.html', prediction_text=f'Estimated House Price: {result_text}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, port=5000)