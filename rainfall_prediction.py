import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("Rainfall.csv")

# Clean column names
data.columns = data.columns.str.strip()

# Drop unnecessary column
if "day" in data.columns:
    data = data.drop(columns=["day"])

# Fill missing values
data["winddirection"] = data["winddirection"].fillna(data["winddirection"].mode()[0])
data["windspeed"] = data["windspeed"].fillna(data["windspeed"].median())

# Encode target variable
data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})

# Drop highly correlated features
data = data.drop(columns=["maxtemp", "temparature", "mintemp"], errors='ignore')

# Balance the dataset
df_majority = data[data["rainfall"] == 1]
df_minority = data[data["rainfall"] == 0]
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
df_downsampled = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42)

# Split features and target
X = df_downsampled.drop(columns=["rainfall"])
y = df_downsampled["rainfall"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest with GridSearch
rf_model = RandomForestClassifier(random_state=42)
param_grid_rf = {
    "n_estimators": [100],
    "max_features": ["sqrt"],
    "max_depth": [None],
    "min_samples_split": [2],
    "min_samples_leaf": [1]
}

grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=1)
grid_search_rf.fit(X_train, y_train)

# Save the best model and feature names
best_rf_model = grid_search_rf.best_estimator_
model_data = {
    "model": best_rf_model,
    "feature_names": X.columns.tolist()
}

with open("rainfall_prediction_machine.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("✅ Model retrained and saved successfully as 'rainfall_prediction_machine.pkl'")
