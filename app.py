from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model file
model = pickle.load(open('attrition_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the webpage
    # Get user inputs from the webpage
    age = float(request.form['age'])
    monthly_income = float(request.form['monthly_income'])
    distance = float(request.form['distance_from_home'])
    job_satisfaction = float(request.form['job_satisfaction'])
    years_at_company = float(request.form['years_at_company'])

    # Send inputs to model
    features = np.array([[age, monthly_income, distance, job_satisfaction, years_at_company]])
    prediction = model.predict(features)[0]
    
    if prediction == 1:
        result = "High Risk: Employee is likely to LEAVE"
    else:
        result = "Low Risk: Employee is likely to STAY"
        
    return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == "__main__":
    app.run(debug=True)
