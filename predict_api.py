import pickle as pkl
import sys
import traceback

import pandas as pd
from flask import Flask, request, jsonify

from src.predict import Predict
from src.preprocess import Preprocessing
from src.build_features import BuildFeatures
from src.train import Train

app = Flask(__name__)


@app.route("/")
def hello():
    return "Welcome to titanic survival check!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Getting json file from post request
        json_ = request.json

        if json_ is None:
            return jsonify({"message": "text not found"})
        else:
            # Json to data frame
            data = pd.DataFrame(json_)

            # Running prediction based on the model
            predict_outcome = Predict()
            prediction = predict_outcome.execute(data, saved_model)

            # prediction = list(model.predict(data))


            return jsonify({'prediction': str(prediction)})

    except:

        return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':

    # Reading initial file
    train_data = pd.read_csv("data/initial_data/train.csv", sep=";")
    print('Reading initial file')

    # Preprocessing
    preprocess = Preprocessing()
    train_data = preprocess.execute(train_data)
    print('Preprocessing')

    # Building features
    build_features = BuildFeatures()
    train_data = build_features.execute(train_data)
    print('Building features')

    # Training model
    train = Train()
    train.execute(train_data, "data/initial_data/model.pkl")
    print('Training model')

    # Reading the model
    saved_model = "data/initial_data/model.pkl"
    print('Model loaded')

    app.run(debug=True, port=5000)
