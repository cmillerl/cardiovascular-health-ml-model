from flask import Flask, request, jsonify
from flask_cors import CORS
from ml_model import MachineLearningModel
import pandas as pd

app = Flask(__name__)
CORS(app)
ml = MachineLearningModel()


@app.route("/", methods=["GET"])
def index():
    return "Welcome to the Cardiovascular Disease Prediction API"


@app.route("/predict", methods=["POST"])
def predict():

    try:
        userInput = [
            float(request.form[k])
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
            ]
        ]

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
        inputDF = pd.DataFrame([userInput], columns=featureNames)
        prediction = ml.makePrediction(inputDF)

        return jsonify({"prediction": "High Risk" if prediction != 0 else "Low Risk"})

    except Exception as e:
        print("🔥 Prediction error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
