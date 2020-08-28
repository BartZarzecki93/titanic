import pandas as pd
import pickle as pkl

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


class Train:
    def execute(self, data, model):

        # Split the data for training.
        y_train = data["Survived"]
        x_train = data.drop(["Survived"], axis=1)

        # Create a classifier and select scoring methods.
        clf = RandomForestClassifier(n_estimators=10)

        # Fit full model and predict on both train and test.
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_train)
        metric_result = accuracy_score(y_train, prediction)
        classification_result = classification_report(y_train, prediction)

        # Serializing and saving model to file
        model_pickle = open(model, 'wb')
        pkl.dump(clf, model_pickle)
        model_pickle.close()

        # Return metrics and model.
        info_accuracy = "Train accuracy for the model is " + str(metric_result)
        info_classification = "Classification report for the model:\n " + str(classification_result)
        print(info_accuracy)
        print(info_classification)
        print(clf)
        return x_train
