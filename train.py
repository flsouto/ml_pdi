import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib

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

# Save the models
joblib.dump(lower, 'lower_model.joblib')
joblib.dump(median, 'median_model.joblib')
joblib.dump(upper, 'upper_model.joblib')

print("Models trained and saved successfully.")