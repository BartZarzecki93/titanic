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

    embarked_dict = {}
    embarked_dict_values = 0

    for i in df.Embarked:
        if i in embarked_dict.keys():
            pass
        else:
            embarked_dict_values = embarked_dict_values + 1
            embarked_dict[i] = embarked_dict_values

    for i in embarked_dict.keys():
        df["Embarked"].replace(i, embarked_dict[i], inplace=True)

    # Family Size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Is Alone?
    df["IsAlone"] = 0
    df.loc[df["FamilySize"] == 1, "IsAlone"] = 1

    # saving new file with preprocessed data
    df.to_csv(output_file)


execute("/Users/bartoszzarzecki/titanic/data/train_new1.csv", "/Users/bartoszzarzecki/titanic/data/clean1.csv")
