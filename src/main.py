import pandas as pd

from src.build_features import BuildFeatures
from src.predict import Predict
from src.preprocess import Preprocessing
from src.train import Train


def titanic_test(input_file):
    # Model path
    model = "../data/model.pkl"

    # Reading initial file
    df = pd.read_csv(input_file, sep=";")

    # Preprocessing
    preprocess = Preprocessing()
    df = preprocess.execute(df)

    # Building features
    build_features = BuildFeatures()
    df = build_features.execute(df)

    # Training model
    train = Train()
    train.execute(df, model)

    # Predicting the output based on the trained model
    predict = Predict()
    predict.execute("../data/val.csv", model)


titanic_test("../data/train.csv")
