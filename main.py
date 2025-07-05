import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Load data from CSV
data = pd.read_csv("task_data.csv")

X = data.drop(columns='time_spent')
y = data['time_spent']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train quantile regressors
lower = GradientBoostingRegressor(loss='quantile', alpha=0.1)
median = GradientBoostingRegressor(loss='quantile', alpha=0.5)
upper = GradientBoostingRegressor(loss='quantile', alpha=0.9)

lower.fit(X_train, y_train)
median.fit(X_train, y_train)
upper.fit(X_train, y_train)

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

