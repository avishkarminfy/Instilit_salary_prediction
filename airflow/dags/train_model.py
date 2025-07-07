import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import joblib

# === Load Data ===
df = pd.read_csv("D:/capstonedata/Software_Salaries.csv")
df.drop_duplicates(inplace=True)
df.drop(columns=['education', 'skills'], inplace=True)

# === Clean Job Titles ===
def clean_job_titles(df):
    mapping = {
        'Sofware Engneer': 'Software Engineer',
        'Software Engr': 'Software Engineer',
        'Softwre Engineer': 'Software Engineer',
        'Dt Scientist': 'Data Scientist',
        'Data Scienist': 'Data Scientist',
        'Data Scntist': 'Data Scientist',
        'ML Engr': 'Machine Learning Engineer',
        'ML Enginer': 'Machine Learning Engineer',
        'Machine Learning Engr': 'Machine Learning Engineer',
        'Software Engr': 'Software Engineer',
    }
    df['job_title'] = df['job_title'].replace(mapping)
    df['job_title'] = df['job_title'].str.title().str.strip()
    return df

df = clean_job_titles(df)
df['experience_level'] = df['experience_level'].fillna(df['experience_level'].mode()[0])
df['employment_type'] = df['employment_type'].fillna(df['employment_type'].mode()[0])
df.drop(columns=['total_salary', 'salary_in_usd', 'conversion_rate'], inplace=True)

# === Remove Outliers ===
def remove_outliers_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

numerical_outliers = ['base_salary', 'bonus', 'stock_options', 'adjusted_total_usd']
df = remove_outliers_iqr(df, numerical_outliers)

# === Split and preprocess ===
TARGET = 'adjusted_total_usd'
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, objective='reg:squarederror', random_state=42)
}

mlflow.set_experiment("salary_prediction_pipeline")
best_model = None
best_score = -np.inf
best_pipeline = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
        mlflow.log_metric("r2", r2_score(y_test, y_pred))
        mlflow.sklearn.log_model(pipeline, name)

        if r2_score(y_test, y_pred) > best_score:
            best_score = r2_score(y_test, y_pred)
            best_model = model
            best_pipeline = pipeline
            best_name = name

# Save best pipeline locally
joblib.dump(best_pipeline, "best_pipeline.pkl")
