import pandas as pd


def execute(input_file, output_file):
    """Builds features

    Args:
        input_file (str): input file.
        output_file (str): output file.
    """

    # Reading preprocessed file
    df = pd.read_csv(input_file)

    # Family Size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Is Alone?
    df["IsAlone"] = 0
    df.loc[df["FamilySize"] == 1, "IsAlone"] = 1

    # Fare
    df['Fare_bin'] = pd.cut(df['Fare'], bins=[0, 7.91, 14.45, 31, 120], labels=['Low_fare', 'median_fare',
                                                                                'Average_fare', 'high_fare'])
    # Title
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Age
    df['Age_bin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 120],
                           labels=['Children', 'Teenage', 'Adult', 'Elder'])
    # Delete unnecessary columns
    df = df.drop(['FamilySize', 'Name', 'Age', "Fare"], axis=1)

    # Create Dummies
    df = pd.get_dummies(df, columns=["Sex", "Age_bin", "Title", "Fare_bin", "Embarked"],
                        prefix=["Sex", "Age_type", "Title", "Fare_type", "Embarked_type"])
    # saving new file with preprocessed data
    df.to_csv(output_file, index=False)
