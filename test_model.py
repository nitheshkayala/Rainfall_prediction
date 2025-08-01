import pickle
import pandas as pd

# Load the retrained model and feature names
with open("rainfall_prediction_machine.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
features = model_data["feature_names"]

# Sample test input — make sure the order and count matches your feature list
# Example values: pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed
sample_input = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
input_df = pd.DataFrame([sample_input], columns=features)

# Make prediction
prediction = model.predict(input_df)[0]
result = "Rainfall" if prediction == 1 else "No Rainfall"

print("🧪 Test Result:", result)
