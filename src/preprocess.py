import pandas as pd


def execute(input_file, output_file):

    # Reading initial file
    data = pd.read_csv(input_file, sep=";")

    # Dropping unuseful columns
    cols = ['Name', 'Ticket', 'Cabin']
    data = data.drop(cols, axis=1)

    # Dropping rows with missing values
    data = data.dropna()

    # Saving new file with preprocessed data
    data.to_csv(output_file, header=True, index=False)