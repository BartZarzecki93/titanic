import pandas as pd

from src.build_features import BuildFeatures
from src.predict import Predict
from src.preprocess import Preprocessing
from src.train import Train


def titanic_survival_test(input_train_data, input_val_data, model):

    # Reading initial file
    train_data = pd.read_csv(input_train_data, sep=";")
    val_data = pd.read_csv(input_val_data, sep=";")

    # Preprocessing
    preprocess = Preprocessing()
    train_data = preprocess.execute(train_data)

    # Building features
    build_features = BuildFeatures()
    train_data = build_features.execute(train_data)

    # Training model
    train = Train()
    train.execute(train_data, model)

    # Predicting the output based on the trained model
    predict = Predict()
    predict.execute(val_data, model)


titanic_survival_test("data/initial_data/train.csv", "data/initial_data/val.csv", "data/initial_data/model.pkl")
