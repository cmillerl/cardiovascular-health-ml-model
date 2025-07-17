"""
Usage: py app.py

Flask back-end REST API for cardiovascular disease prediction.

API endpoints:
- GET / - Returns a welcome message
- POST /predict - Submit patient data and receive a prediction from the cardiovascular disease prediction machine learning model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ml_model import MachineLearningModel
import pandas as pd

# Initialize Flask app and machine learning model.
# CORS is used to allow cross-origin requests.
app = Flask(__name__)
CORS(app)
ml = MachineLearningModel()


@app.route("/", methods=["GET"])
def index():
    """
    Welcome endpoint for the REST API

    - GET / - Returns a welcome message
    """
    return "Welcome to the Cardiovascular Disease Prediction API"


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict cardiovascular disease risk based on patient data.

    Expects the following form data:

    - age: patient age in years (float)
    - gender: patient assigned gender at birth (1 = woman, 2 = man)
    - height: patient height in inches (float)
    - weight: patient weight in pounds (float)
    - ap_hi: patient systolic blood pressure (float)
    - ap_lo: patient diastolic blood pressure (float)
    - cholesterol: patient cholesterol level (1 = normal, 2 = above normal, 3 = well above normal)
    - gluc: patient glucose level (1 = normal, 2 = above normal, 3 = well above normal)
    - smoke: patient smoking status (0 = no, 1 = yes)
    - alco: patient alcohol consumption status (0 = no, 1 = yes)
    - active: patient physical activity status (0 = no, 1 = yes)

    Returns a JSON response with the prediction result:
    - prediction: "High Risk" or "Low Risk" based on the model's prediction (0 = low risk, 1 = high risk)
    """

    try:
        # Extract user input from the request form.
        # Convert form data to a list of floats.
        # Ensure the form keys match the expected input features.
        userInput = []
        for k in [
            "age",
            "gender",
            "height",
            "weight",
            "ap_hi",
            "ap_lo",
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
        ]:
            if k == "height":
                userInput.append(float(request.form[k]) / 12)  # Convert inches to feet.
            else:
                userInput.append(float(request.form[k]))

        # Define the feature names and create a DataFrame for prediction.
        # The feature names must match the model's expected input.
        featureNames = [
            "age",
            "gender",
            "height",
            "weight",
            "ap_hi",
            "ap_lo",
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
        ]

        # Create a DataFrame with the user input.
        # The DataFrame must have the same structure as the training data used for the model.
        inputDF = pd.DataFrame([userInput], columns=featureNames)

        # Make a prediction using the trained machine learning model.
        # 0 for Low Risk, 1 for High Risk.
        prediction = ml.makePrediction(inputDF)

        # Convert the numerical prediction to a string.
        # Return the prediction result as a JSON response.
        return jsonify({"prediction": "High Risk" if prediction != 0 else "Low Risk"})

    except Exception as e:
        # Handle and log any errors that occur during prediction.
        print("🔥 Prediction error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Production deployment configuration
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
