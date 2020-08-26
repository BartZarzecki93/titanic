import pandas as pd
import pickle as pkl

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train(input_file, model):
    # Reading file after build_features
    df = pd.read_csv(input_file)

    # Split the data for training.
    y_train = df["Survived"]
    x_train = df.drop(["Survived"], axis=1)

    # Create a classifier and select scoring methods.
    clf = RandomForestClassifier(n_estimators=10)

    # Fit full model and predict on both train and test.
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_train)
    metric_name = "train_accuracy"
    metric_result = accuracy_score(y_train, prediction)

    # Serializing and saving to file
    model_pickle = open(model, 'wb')
    pkl.dump(clf, model_pickle)
    model_pickle.close()

    # Return metrics and model.
    info = ""
    info = info + metric_name
    info = info + " for the model is "
    info = info + str(metric_result)
    print(info)
