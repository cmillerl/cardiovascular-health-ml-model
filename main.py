"""
Usage: py main.py
"""

import ml_model


class Main:
    """
    Main application to initialize and run the machine learning model

    If inside the data folder scaler.pkl, cardiovascular_model.pkl, and/or statistics.json don't exist, run this model to create and populate them.

    If inside the statistics.json file the highest accuracy doesn't exist, run this model to create and populate it.
    """

    def __init__(self):
        # Initialize the main application with a machine learning model instance.
        self.model = ml_model.MachineLearningModel()

    def run(self):
        # Run the machine learning model to perform training, evaluation, and saving of the model.
        self.model.runModel()


if __name__ == "__main__":
    # Create an instance of the Main class and run the application.
    main = Main()
    main.run()
