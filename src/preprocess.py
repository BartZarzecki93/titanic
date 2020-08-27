import pandas as pd
from src.build_features import BuildFeatures


class Preprocessing:
    def execute(self, data):

        # Dropping unuseful columns
        cols = ['Ticket', "Cabin", "PassengerId"]
        data = data.drop(cols, axis=1)

        # Dropping rows with missing values
        data = data.dropna()

        # Building features
        build_features = BuildFeatures()
        data = build_features.execute(data)

        return data
