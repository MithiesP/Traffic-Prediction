import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib  # Import joblib to load the model


app = Flask(__name__)

# Load the trained model from the pickle file
regressor = joblib.load('traffic_model.pkl')
scaler = joblib.load('scaler_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    day = int(request.form['day'])
    zone = int(request.form['zone'])
    weather = int(request.form['weather'])
    temperature = float(request.form['temperature'])

    # Preprocess the input data
    input_data = np.array([[day, zone, weather, temperature]])
    # Load the StandardScaler and fit it to the data
    input_data = scaler.transform(input_data)

    # Make a prediction using the trained model
    prediction = regressor.predict(input_data)
    
    # Round the prediction based on your criteria
    if prediction[0] < 2.5:
        prediction = np.round(prediction - 0.5)
    else:
        prediction = np.round(prediction + 0.5)

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
