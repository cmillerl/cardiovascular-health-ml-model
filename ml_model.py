import json
import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class MachineLearningModel:
    """
    This class handles the loading, preprocessing, training, and saving of the logistic regression model for predicting cardiovascular disease risk.
    """

    def __init__(self):

        # Define the path to the CSV file, the statistics file, and the scaled data file.
        self.dataFilePath = "data/cardiovascular-data.csv"
        self.statisticsPath = "data/statistics.json"
        self.scaledDataPath = "data/scaler.pkl"

        # Retrieves the highest accuracy achieved by the model.
        self.highestAccuracy = self.getHighestAccuracy()

        # Read the CSV file using a semicolon as the delimiter (delimiter = the separator between values).
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

        # Filter out rows with invalid values.
        # Accept only values 0-3 for categorical features.
        # 1=normal, 2=above normal, 3=well above normal for cholesterol and glucose.
        # 0-1 for binary features (0=no, 1=yes for smoke/alcohol/active/cardio).
        self.acceptedValues = {0, 1, 2, 3}
        self.df = self.df[self.df["cholesterol"].isin(self.acceptedValues)]
        self.df = self.df[self.df["gluc"].isin(self.acceptedValues)]
        self.df = self.df[self.df["smoke"].isin(self.acceptedValues)]
        self.df = self.df[self.df["alco"].isin(self.acceptedValues)]
        self.df = self.df[self.df["active"].isin(self.acceptedValues)]
        self.df = self.df[self.df["cardio"].isin(self.acceptedValues)]

        # Convert age from days to years.
        self.df["age"] = self.df["age"] / 365.25
        # Convert weight from kg to lbs.
        self.df["weight"] = self.df["weight"] * 2.20462
        # Convert height from cm to inches, then to feet (cm * 0.393701 = inches, inches / 12 = feet).
        self.df["height"] = (self.df["height"] * 0.393701) / 12

        # Filter out unrealistic physical attribute values.
        # Height: 1-8 feet, Weight: 1-1000 lbs, Blood pressure (ap_hi = systolic, ap_lo = diastolic): 1-500 mmHg.
        self.df = self.df[(self.df["height"] <= 8.0) & (self.df["height"] >= 1.0)]
        self.df = self.df[(self.df["weight"] > 1.0) & (self.df["weight"] < 1000.0)]
        self.df = self.df[(self.df["ap_hi"] <= 500) & (self.df["ap_hi"] > 0)]
        self.df = self.df[(self.df["ap_lo"] <= 500) & (self.df["ap_lo"] > 0)]

    def getHighestAccuracy(self):
        """
        Reads the statistics JSON file to get the highest accuracy achieved by the model.
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
        except (FileNotFoundError):
            print("Error reading the JSON statistics file.")
            return 0.0

    def saveHighestAccuracy(self, accuracy):
        """
        Saves the highest accuracy achieved by the model to the statistics JSON file.
        """
        try:
            with open(self.statisticsPath, "w") as file:
                json.dump({"highestAccuracy": accuracy}, file, indent=4)
        except (FileNotFoundError):
            print("Error writing to the JSON statistics file.")

    def runModel(self):
        """
        Trains and evaluates a logistic regression model for cardiovascular disease prediction.

        This method performs:
        - Feature scaling using StandardScaler
        - Train/test split (80/20)
        - Model training with logistic regression
        - Accuracy evaluation

        The trained model and scaler are saved to the data folder for later use in predictions.
        """

        scaler = StandardScaler()

        # Separate the features and the target variable.
        # X = the input features (independent variables) (e.g., age, height, weight).
        # y = the target variable (dependent variable) (risks of cardiovascular disease (0 = low risk, 1 = high risk)).
        X = self.df.drop("cardio", axis=1)
        y = self.df["cardio"]

        # Split the dataset into training and testing sets.
        # Split data into 80% training and 20% testing.
        # Setting random_state=42 ensures reproducibility of the results.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # fmt: skip

        # Store the training and testing data as variables for visualizations.
        self.X_test = X_test
        self.y_test = y_test

        # Transform the data to ensure all features are on the same scale.
        # This is important for models like logistic regression.
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save the scaler to a file for later use.
        joblib.dump(scaler, self.scaledDataPath)

        # Train the logistic regression model with a maximum of 2,000 iterations.
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train_scaled, y_train)

        # Predict cardiovascular disease risk on the test set and calculate accuracy by comparing the predicted values against the actual values (outcomes).
        y_pred = model.predict(X_test_scaled)
        accuracy = round(accuracy_score(y_test, y_pred), 3)
        print(f"Accuracy of the logistic regression model: {accuracy * 100}%")

        # Check if the accuracy of the model is higher than the previously saved highest accuracy.
        # If it is, save the new accuracy and the model to a file.
        if accuracy > self.highestAccuracy:
            self.saveHighestAccuracy(accuracy)
            joblib.dump(model, "data/cardiovascular_model.pkl")
            print("Model accuracy improved and saved.")
        else:
            print("No improvement in model accuracy.")

        # Check if the model file already exists.
        # If it does not exist, save the model.
        if not os.path.exists("data/cardiovascular_model.pkl"):
            print("Model file not found, saving the model.")
            joblib.dump(model, "data/cardiovascular_model.pkl")

        # Store the model and predictions as a variable for visualizations.
        self.y_pred = y_pred

    def makePrediction(self, userInput):
        """
        Make a prediction based on the user input.

        Input data must contain all the required patient features.

        If the model file does not exist, it will run the model to create it.

        If the scaler file does not exist, it will run the model to create it.

        Returns the predicted risk of cardiovascular disease (0 = low risk, 1 = high risk).
        """

        self.modelPath = "data/cardiovascular_model.pkl"

        if not os.path.exists(self.modelPath):
            print("Model training data not found, running the model to create it.")
            self.runModel()

        if not os.path.exists(self.scaledDataPath):
            print("Scaler data not found, running the model to create it.")
            self.runModel()

        model = joblib.load(self.modelPath)
        scaler = joblib.load(self.scaledDataPath)
        scaledInput = scaler.transform(userInput)
        prediction = model.predict(scaledInput)
        return prediction[0]
    
    def plotConfusionMatrix(self):
        """
        Plots the confusion matrix.

        The confusion matrix provides a more detailed breakdown of the model by displaying the numbers of true and false positives, as well as true and false negatives.
        """

        # Create the confusion matrix.
        cm = confusion_matrix(self.y_test, self.y_pred)

        # Create the visualization.
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                     xticklabels=['Low Risk CVD', 'High Risk CVD'], yticklabels=['Low Risk CVD', 'High Risk CVD'])
        plt.title("Confusion Matrix")
        plt.ylabel("Actual Values")
        plt.xlabel("Predicted Values")
        plt.show()

    def plotDataDistribution(self):
        """
        Plots a data distribution chart (histogram).

        Used to visually compare the distribution of features (health indicators) before and after scaling.
        """

        # The scaling process needs to be recreated for this visualization to properly display the data before and after scaling.
        X = self.df.drop("cardio", axis=1)
        y = self.df["cardio"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)

        # Plot histograms for each feature in the dataset.
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        axes = axes.ravel()

        for i, column in enumerate(X.columns):
            # Data before scaling.
            axes[i].hist(X_train[column], bins=30, alpha=0.7, label='Original Data', color='blue')
            # Data after scaling.
            axes[i].hist(X_train_scaled_df[column], bins=30, alpha=0.7, label='Scaled Data', color='orange')

            axes[i].set_title(column.title().lower())
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        # Remove the empty histogram.
        axes[11].axis('off')

        plt.tight_layout()
        plt.show()


    def printClassificationReport(self):
        """
        Prints the classification report of the model.

        The classification report shows that the average of our model’s precision, recall, and F1 score.
        """
        print("Classification Report\n")
        report = classification_report(self.y_test, self.y_pred, target_names=["Low Risk", "High Risk"])
        print(report)

    def plotCorrelationMap(self):
        """
        Plots a correlation heatmap.
        
        The correlation heatmap shows the correlation value between features (health indicators) and the target variable (risk of CVD)
        """

        # Calculate the correlation matrix.
        correlation_matrix = self.df.corr()

        # Create the correlation heatmap.
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='Purples', square=True, cbar_kws={"shrink": .8})
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
