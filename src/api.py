import pickle as pkl
import sys
import traceback

import pandas as pd
from flask import Flask, request, jsonify

from src.preprocess import Preprocessing
from src.build_features import BuildFeatures

app = Flask(__name__)


@app.route("/")
def hello():
    return "Welcome to titanic survival check!"


@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            json_ = request.json
            print(json_)
            if json_ is None:
                return jsonify({"message": "text not found"})
            else:
                # Json to data frame
                data = pd.DataFrame(json_)
                print(data)

                # Preprocessing with building features
                preprocess = Preprocessing()
                data = preprocess.execute(json_)

                # Building features
                build_features = BuildFeatures()
                data = build_features.execute(data)

                data = data.drop(["Survived"], axis=1)

                print(model.predict(data))
                prediction = list(model.predict(data))
                return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return 'There is no model to use'


if __name__ == '__main__':
    try:
        # Command-line input
        port = int(sys.argv[1])
    except:
        # If you don't provide any port the port will be set to 12345
        port = 12345

    # Loading the model
    saved_model = "../data/model.pkl"
    model_unpickle = open(saved_model, 'rb')
    model = pkl.load(model_unpickle)
    model_unpickle.close()
    print('Model loaded')

    app.run(debug=True)
