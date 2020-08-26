import pandas as pd
import pickle as pkl


def predict(input_file, model):
    # Reading file for prediction that is preprocessed
    df = pd.read_csv(input_file)

    target = df["Survived"]
    df = df.drop(["Survived"], axis=1)

    # Unpickling files
    model_unpickle = open(model, 'rb')
    model = pkl.load(model_unpickle)
    model_unpickle.close()

    # Run prediction based on the model
    predictions = model.predict(df)

    # Reassign target (if it was present) and predictions.
    df["Prediction"] = predictions
    df["Target"] = target

    ok = 0
    for i in df.iterrows():
        if i[1]["Target"] == i[1]["Prediction"]:
            ok = ok + 1

    print("accuracy is", ok / df.shape[0])
