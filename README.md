# Instilit_salary_prediction

This project implements an **end-to-end MLOps workflow** for predicting software engineering salaries using multiple ML models with tools like **MLflow**, **Apache Airflow**, **Evidently AI**, and **Flask**.

---

## Tech Stack

| Layer            | Tools Used                                                                 |
|------------------|----------------------------------------------------------------------------|
| Language         | Python                                                                     |
| Model Training   | scikit-learn, XGBoost, pandas, numpy                                        |
| Experimentation  | **MLflow**                                                                 |
| Drift Detection  | **Evidently AI**                                                           |
| Automation       | **Apache Airflow**                                                         |
| Deployment       | **Flask**                                                                  |
| Database         | PostgreSQL                                                                 |

---

## Project Structure

---

## MLflow Experiment Tracking

All model runs are tracked with:
- **Metrics**: R² Score, MAE, MSE
- **Artifacts**: Drift reports, models
- **Parameters**: Model type, pipeline steps

![image](https://github.com/user-attachments/assets/5c8272e2-3d28-40b8-8788-ce22b107b852)

---

## Model Registry

After evaluating all models, the **best performing model (based on R²)** is automatically:
- Logged in MLflow
- Registered in the Model Registry as `Best_Salary_Predictor`

![image](https://github.com/user-attachments/assets/48c59c6f-cac4-44a9-a4b7-6730414f0322)

---

## Data Drift Detection

Drift detection is performed using **Evidently AI** between training and test datasets.
The HTML report is saved as an artifact in MLflow.

![image](https://github.com/user-attachments/assets/c5de31f0-c7e2-471a-8631-3009e7843003)

---

## Apache Airflow DAG

Apache Airflow orchestrates the complete ML pipeline:
1. Drift Detection
2. Model Training
3. Model Registration

The DAG runs periodically or can be triggered manually.

![image](https://github.com/user-attachments/assets/45a84531-2dfa-452c-965f-4218237b8089)

![image](https://github.com/user-attachments/assets/b376b7bf-fbd8-4298-8c8e-094786858552)

---

## Flask Web UI

A Flask-based user interface allows users to:
- Input features like job title, experience, location, etc.
- Submit salary details
- Get real-time salary predictions in **USD**

![image](https://github.com/user-attachments/assets/dd853cdf-a77f-451f-92aa-3a39a81c4e8f)

![image](https://github.com/user-attachments/assets/40f0c52f-af3c-4ecb-ae17-f0a418a31858)

---

## Best Model Metrics

The following model was selected and registered as the best:

| Model         | R² Score | MAE     | MSE           |
|---------------|----------|---------|----------------|
| **XGBoost**   | 0.9997   | 687.13  | 899,145.16     |

---

## Explainability

SHAP plots were generated for model interpretation and logged as part of the evaluation process (for tree-based models only).

![image](https://github.com/user-attachments/assets/86815ca3-0ec1-4767-bc37-7a96cd8a176f)


---












