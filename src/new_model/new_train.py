import pandas as pd
import pickle as pkl

from src.preprocess import Preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def train(input_file, model):
    # Reading file after build_features
    df = pd.read_csv(input_file, sep=";")

    # Preprocessing with building features
    preprocess = Preprocessing()
    df = preprocess.execute(df)

    # Split the data for training.
    y_train = df["Survived"]
    x_train = df.drop(["Survived"], axis=1)

    # Create SVC model
    clf = SVC()
    # Fit full model and predict on both train and test.
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_train)
    metric_name = "Train accuracy"
    metric_result = accuracy_score(y_train, prediction)
    classification_result = classification_report(y_train, prediction)

    # Serializing and saving to file
    model_pickle = open(model, 'wb')
    pkl.dump(clf, model_pickle)
    model_pickle.close()

    # Return metrics and model.
    info_accuracy = "Train accuracy for the model is " + str(metric_result)
    info_classification = "Classification report for the model:\n " + str(classification_result)
    print(info_accuracy)
    print(info_classification)
train("../../data/train.csv", "../../data/model_new.pkl")