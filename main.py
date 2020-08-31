import traceback
import pandas as pd

from flask import Flask, request, jsonify
from src.predict import Predict
from src.preprocess import Preprocessing
from src.build_features import BuildFeatures
from src.train import Train

app = Flask(__name__)


# Welcome message
@app.route("/")
def hello():
    return "Welcome to titanic survival check!"


# Prediction URL
@app.route('/predict', methods=['POST'])
def predict():
    if saved_model:
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

                # Concat data for more detailed json result
                concat_data = pd.concat([data, prediction.reindex(data.index)], axis=1)
                predictions_results = []
                for row in concat_data.iterrows():
                    if row[1]["Prediction"] == 1:
                        predictions_results.append({"Name": row[1]["Name"], "Result": "Would survive the crash"})
                    else:
                        predictions_results.append({"Name": row[1]["Name"], "Result": "Would not survive the crash"})

                return jsonify(predictions_results)

        except:
            # Return the error
            return jsonify({traceback.format_exc()})
    else:
        print('There is no model')
        return 'No model to use'


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

    app.run(debug=True, host='0.0.0.0')
