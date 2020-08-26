import pandas as pd
import pickle as pkl

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score


def train(input_file, model):
    # Reading file after build_features
    df = pd.read_csv(input_file)

    # Split the data for training.
    y_train = df["Survived"]
    x_train = df.drop(["Survived", "PassengerId"], axis=1)

    # Create SVC model
    clf = SVC()
    # Fit full model and predict on both train and test.
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_train)
    metric_name = "Train accuracy"
    metric_result = accuracy_score(y_train, prediction)

    # Serializing and saving to file
    model_pickle = open(model, 'wb')
    pkl.dump(clf, model_pickle)
    model_pickle.close()

    # Return metrics and model.
    info = metric_name + " for the model is "
    info = info + str(metric_result)
    print(info)
