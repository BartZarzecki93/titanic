import pandas as pd


# import numpy as np


def execute(input_file, output_file):

    data = pd.read_csv(input_file, sep=";")

    # dropping unuseful columns
    cols = ['Name', 'Ticket', 'Cabin']
    data = data.drop(cols, axis=1)

    # dropping rows with missing values
    data = data.dropna()

    # saving new file with preprocessed data
    data.to_csv(output_file)


execute("/Users/bartoszzarzecki/titanic/data/train.csv", "/Users/bartoszzarzecki/titanic/data/train_new1.csv")
