# detect_drift.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load old and new data
old_data = pd.read_csv("dags/data/old_data.csv")
new_data = pd.read_csv("dags/data/new_data.csv")

# Drop target column if exists
for df in [old_data, new_data]:
    for col in ['adjusted_total_usd', 'total_salary', 'salary_in_usd']:
        df.drop(columns=col, errors='ignore', inplace=True)

# Generate drift report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=old_data, current_data=new_data)

# Save report
report.save_html("dags/reports/drift_report.html")

# Optional: save drift result as flag
drift_result = report.as_dict()
with open("dags/flags/drift_detected.txt", "w") as f:
    if drift_result['metrics'][0]['result']['dataset_drift']:
        f.write("True")
    else:
        f.write("False")

print("✅ Drift detection completed.")
