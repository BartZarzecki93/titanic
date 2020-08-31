import pickle as pkl

from src.build_features import BuildFeatures
from src.preprocess import Preprocessing


class Predict:
    def execute(self, input_data, model):

        # Preprocessing with building features
        preprocess = Preprocessing()
        data = preprocess.execute(input_data)

        # Building features
        build_features = BuildFeatures()
        data = build_features.execute(data)

        # Getting the target and preparing data for prediction
        target = data["Survived"]
        data = data.drop(["Survived"], axis=1)

        # Unpickling files
        model_unpickle = open(model, 'rb')
        model = pkl.load(model_unpickle)
        model_unpickle.close()

        # Run prediction based on the created model
        predictions = model.predict(data)

        # Reassign target (if it was present) and predictions.
        data["Prediction"] = predictions
        data["Target"] = target

        match_count = 0
        for row in data.iterrows():
            if row[1]["Target"] == row[1]["Prediction"]:
                match_count = match_count + 1

        print("Accuracy is", match_count / data.shape[0])
        return data
