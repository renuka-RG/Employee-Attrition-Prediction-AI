from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the webpage
    age = float(request.form['age'])
    monthly_income = float(request.form['monthly_income'])
    distance = float(request.form['distance_from_home'])
    job_satisfaction = float(request.form['job_satisfaction'])
    years_at_company = float(request.form['years_at_company'])

    # Send inputs to model
    features = np.array([[age, monthly_income, distance, job_satisfaction, years_at_company]])
    prediction = model.predict(features)[0]

    return render_template('index.html', prediction_text=f'Employee Attrition Prediction: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
