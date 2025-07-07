# app.py
import pandas as pd
import mlflow.pyfunc
from flask import Flask, request, render_template

mlflow.set_tracking_uri("http://127.0.0.1:5000")

app = Flask(__name__)

# Load the best model from MLflow registry
MODEL_NAME = "Best_Salary_Predictor"
MODEL_STAGE_OR_VERSION = "1"
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE_OR_VERSION}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'job_title': request.form['job_title'],
            'experience_level': request.form['experience_level'],
            'employment_type': request.form['employment_type'],
            'company_size': request.form['company_size'],
            'company_location': request.form['company_location'],
            'salary_currency': request.form['salary_currency'],
            'currency': request.form['currency'],
            'remote_ratio': int(request.form['remote_ratio']),
            'years_experience': float(request.form['years_experience']),
            'base_salary': float(request.form['base_salary']),
            'bonus': float(request.form['bonus']),
            'stock_options': float(request.form['stock_options'])
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        return render_template('index.html', prediction=f"{prediction:,.2f}")

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True,port=8000)
