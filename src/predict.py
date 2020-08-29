import pickle as pkl
import pandas as pd

from src.build_features import BuildFeatures
from src.preprocess import Preprocessing


class Predict:
    def execute(self, input_file, model):

        # Reading file for prediction that is preprocessed
        df = pd.read_csv(input_file, sep=";")

        # Preprocessing with building features
        preprocess = Preprocessing()
        df = preprocess.execute(df)

        # Building features
        build_features = BuildFeatures()
        df = build_features.execute(df)

        # Getting the target and preparing data for prediction
        target = df["Survived"]
        df = df.drop(["Survived"], axis=1)

        # Unpickling files
        model_unpickle = open(model, 'rb')
        model = pkl.load(model_unpickle)
        model_unpickle.close()

        # Run prediction based on the created model
        predictions = model.predict(df)

        # Reassign target (if it was present) and predictions.
        df["Prediction"] = predictions
        df["Target"] = target

        match_count = 0
        for row in df.iterrows():
            if row[1]["Target"] == row[1]["Prediction"]:
                match_count = match_count + 1

        print("Accuracy is", match_count / df.shape[0])
        return df
