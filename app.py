from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
with open("rainfall_prediction_machine.pkl", "rb") as file:
    model_data = pickle.load(file)
model = model_data["model"]
features = model_data["feature_names"]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        values = [float(request.form.get(f)) for f in features]
        input_df = pd.DataFrame([values], columns=features)
        
        # Predict
        prediction = model.predict(input_df)[0]
        result = "Rainfall" if prediction == 1 else "No Rainfall"
        return render_template("index.html", prediction_text=f"Prediction: {result}")
    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
