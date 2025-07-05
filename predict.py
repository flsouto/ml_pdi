import pandas as pd
import joblib

# Load the models
lower = joblib.load('lower_model.joblib')
median = joblib.load('median_model.joblib')
upper = joblib.load('upper_model.joblib')

# Predict for a new task
X_new = pd.DataFrame([{
    'has_backend': 1,
    'has_frontend': 1,
    'project_complexity': 2,
    'developer_seniority': 2
}])

min_time = lower.predict(X_new)[0]
median_time = median.predict(X_new)[0]
max_time = upper.predict(X_new)[0]

print(f"Estimated time range: {min_time:.1f}h – {max_time:.1f}h (median: {median_time:.1f}h)")