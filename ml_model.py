import json
import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MachineLearningModel:
    def __init__(self):

        # Define the path to the CSV file and the statistics file
        self.dataFilePath = "data/cardiovascular-data.csv"
        self.statisticsPath = "data/statistics.json"

        # Store the highest accuracy achieved by the model
        self.highestAccuracy = self.getHighestAccuracy()

        # Read the CSV file with semicolon as delimiter
        self.df = pd.read_csv(self.dataFilePath, delimiter=";")
        self.df = self.df[
            [
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
                "cardio",
            ]
        ]

        # Filters out rows with missing values in any of the specified columns
        self.acceptedValues = {0, 1, 2, 3}
        self.df = self.df[self.df["cholesterol"].isin(self.acceptedValues)]
        self.df = self.df[self.df["gluc"].isin(self.acceptedValues)]
        self.df = self.df[self.df["smoke"].isin(self.acceptedValues)]
        self.df = self.df[self.df["alco"].isin(self.acceptedValues)]
        self.df = self.df[self.df["active"].isin(self.acceptedValues)]
        self.df = self.df[self.df["cardio"].isin(self.acceptedValues)]

        # Convert age from days to years
        self.df["age"] = self.df["age"] / 365.25
        # Convert weight from kg to lbs
        self.df["weight"] = self.df["weight"] * 2.20462
        # Convert height from cm to inches to feet
        self.df["height"] = self.df["height"] * 0.393701 / 12

        # Filter out unrealistic values
        self.df = self.df[(self.df["height"] <= 8.0) & (self.df["height"] >= 1.0)]
        self.df = self.df[(self.df["ap_hi"] <= 500) & (self.df["ap_hi"] > 0)]
        self.df = self.df[(self.df["ap_lo"] <= 500) & (self.df["ap_lo"] > 0)]

    def getHighestAccuracy(self):
        """
        Reads the statistics file to get the highest accuracy achieved by the model.
        """
        try:
            if os.path.exists(self.statisticsPath):
                with open(self.statisticsPath, "r") as file:
                    data = json.load(file)
                if "highestAccuracy" in data:
                    return data["highestAccuracy"]
                else:
                    return 0.0
            else:
                return 0.0
        except ValueError:
            print("Error reading the JSON statistics file.")
            return 0.0

    def saveHighestAccuracy(self, accuracy):
        """
        Saves the highest accuracy achieved by the model to the statistics file.
        """
        try:
            with open(self.statisticsPath, "w") as file:
                json.dump({"highestAccuracy": accuracy}, file, indent=4)
        except ValueError:
            print("Error writing to the JSON statistics file.")

    def runModel(self):
        """
        Separate the feature and the target variable
        X = what the model will learn from
        y = what the model will predict
        Both based on the risks of cardiovascular disease (0 = no, 1 = yes)
        """
        X = self.df.drop("cardio", axis=1)
        y = self.df["cardio"]

        """
        Split the dataset into training and testing sets
        Using a random state of 42 ensures the numbers generated are reproducible
        Test size of 0.2 means 20% of the data will be used for testing and 80% for training
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # fmt: skip

        """
        Train the logistic regression model with a maximum of 2,000 iterations
        """
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)

        """
        Make predictions on the test set and print the accuracy of the model
        """
        y_pred = model.predict(X_test)
        accuracy = round(accuracy_score(y_test, y_pred), 3)
        print(f"Accuracy of the logistic regression model: {accuracy * 100}%")

        if accuracy > self.highestAccuracy:
            self.saveHighestAccuracy(accuracy)
            joblib.dump(model, "data/cardiovascular_model.pkl")
            print("Model accuracy improved and saved.")
        else:
            print("No improvement in model accuracy.")

    def makePrediction(self, userInput):
        """
        Make a prediction based on the user input
        Input data must contain all the required features
        """
        self.modelPath = "data/cardiovascular_model.pkl"

        if not os.path.exists(self.modelPath):
            print("Model training data not found, running the model to create it.")
            self.runModel()

        model = joblib.load(self.modelPath)
        prediction = model.predict(userInput)
        return prediction[0]
