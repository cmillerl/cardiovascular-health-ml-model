"""
Usage: py main.py
"""

import ml_model
from time import sleep


class Main:
    """
    Main application to initialize and run the machine learning model or visualizations.

    If inside the data folder scaler.pkl, cardiovascular_model.pkl, and/or statistics.json don't exist, run this model to create and populate them.

    If inside the statistics.json file the highest accuracy doesn't exist, run this model to create and populate it.
    """

    def __init__(self):
        self.model = ml_model.MachineLearningModel()
        self.acceptedValues = [1, 2, 3, 4]

    def run(self):
        """
        Run the machine learning model and ask the user if they'd like to see visualizations.

        If the user chooses to see visualizations, it will prompt them to see which visualizations they want to see.

        If the user chooses to run the model, it will run the model and save the results.
        """

        print("Running the machine learning model.")
        sleep(2)
        # Runs the machine learning model.
        self.model.runModel()
        sleep(2)
        
        # Asks the user if they want to see visualizations and stores their answer in a variable.
        userInput = (
            input("Do you want to see visualizations? Type yes or no: ").strip().lower()
        )
        sleep(2)
        # If the user inputs "yes", a list will be printed containing the available visualizations.
        if userInput == "yes":
            print("\nAvailable Visuals")
            print("1. Classification Report")
            print("2. Confusion Matrix")
            print("3. Data Distribution (Histogram)")
            print("4. Correlation Heatmap")
            sleep(2)
            while True:
                try:
                    # Asks the user which visual they want to see and stores their answer in a variable.
                    userChoice = input("\nEnter the number of the visual you want to see (1-4) or type exit: ").strip()
                    sleep(1)
                    # If the user inputs "exit", the program will print a goodbye message and exit.
                    if userChoice == "exit":
                        print("Okay, goodbye!")
                        exit()
                    userChoice = int(userChoice)
                    if userChoice in self.acceptedValues:
                        if userChoice == 1:
                            # Prints the classification report.
                            self.model.printClassificationReport()
                        elif userChoice == 2:
                            # Plots the confusion matrix.
                            self.model.plotConfusionMatrix()
                        elif userChoice == 3:
                            # Plots the data distribution (histogram).
                            self.model.plotDataDistribution()
                        elif userChoice == 4:
                            # Plots the correlation heatmap.
                            self.model.plotCorrelationMap()
                    else:
                        # If the user inputs a number that is not in the accepted values, it will print an error message and then ask the user to input a number again continuously until they input a valid number or type "exit".
                        print("Invalid choice. Please enter a whole number between 1 and 4 or type exit.\n")
                        continue
                except ValueError:
                    print("Invalid input. Please enter a whole number between 1 and 4 or type exit.\n")
                    continue
        else:
            # If the user inputs anything other than "yes", the program will print a goodbye message and exit.
            print("Okay, goodbye!")
            exit()


if __name__ == "__main__":
    # Create an instance of the Main class and run the application.
    main = Main()
    main.run()
