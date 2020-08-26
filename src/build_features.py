import pandas as pd


def execute(input_file, output_file):
    """Builds features

    Args:
        input_file (str): input file.
        output_file (str): output file.
    """

    # Reading preprocessed file
    df = pd.read_csv(input_file)

    # Replacing text with binary in sex
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)

    # Embarked (Switching to int after research (only three letters) )
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Family Size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Is Alone?
    df["IsAlone"] = 0
    df.loc[df["FamilySize"] == 1, "IsAlone"] = 1

    # saving new file with preprocessed data
    df.to_csv(output_file, index=False)
