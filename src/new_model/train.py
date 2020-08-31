import pickle as pkl

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

from src.preprocess import Preprocessing


class Train:
    def execute(self, data, model):

        # Split the data for training.
        y_train = data["Survived"]
        x_train = data.drop(["Survived"], axis=1)

        # Create SVC model
        clf = SVC()

        # Fit full model and predict on both train and test.
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_train)
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

        return x_train, clf
