from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('ridge_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = [
        int(data['year']),
        float(data['present_price']),
        int(data['kms_driven']),
        data['fuel_type'],
        data['seller_type'],
        data['transmission'],
        int(data['owner'])
    ]

    input_df = {
        'Year': [features[0]],
        'Present_Price': [features[1]],
        'Kms_Driven': [features[2]],
        'Fuel_Type': [features[3]],
        'Seller_Type': [features[4]],
        'Transmission': [features[5]],
        'Owner': [features[6]],
    }

    import pandas as pd
    input_df = pd.DataFrame(input_df)
    prediction = model.predict(input_df)[0]

    return render_template('result.html', prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
