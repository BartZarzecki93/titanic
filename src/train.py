import pandas as pd
import pickle as pkl

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Reading file after build_features
df = pd.read_csv("/data/train_bf.csv")

# Split the data for training.
Y_train = df["Survived"]
X_train = df.drop(["Survived"], axis=1)

# Create a classifier and select scoring methods.
clf = RandomForestClassifier(n_estimators=10)

# Fit full model and predict on both train and test.
clf.fit(X_train, Y_train)
prediction = clf.predict(X_train)
metric_name = "train_accuracy"
metric_result = accuracy_score(Y_train, prediction)

# Serializing and saving to file
model_pickle = open("data/model.pkl", 'wb')
pkl.dump(clf, model_pickle)
model_pickle.close()

# Return metrics and model.
info = ""
info = info + metric_name
info = info + " for the model is "
info = info + str(metric_result)
print(info)
