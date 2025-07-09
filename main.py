import ml_model


# Main application to initialize and run the machine learning model
# If an accuracy exists in statistics.json and a cardiovascular_model.pkl file is present, there's no need to run this.
class Main:
    def __init__(self):
        self.model = ml_model.MachineLearningModel()

    def run(self):
        self.model.runModel()


if __name__ == "__main__":
    main = Main()
    main.run()
