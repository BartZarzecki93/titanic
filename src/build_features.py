import pandas as pd


class BuildFeatures:
    def execute(self, data):

        # Building family size column based on SibSp and Parch
        data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

        # Is Alone? questioning if person was alone or no, data retrieved from FamilySize
        data["IsAlone"] = 0
        data.loc[data["FamilySize"] == 1, "IsAlone"] = 1

        # Creating fare categories based on ranges of prices
        data['Fare_bin'] = pd.cut(data['Fare'], bins=[0, 7.91, 14.45, 31, 120], labels=['Low_fare', 'median_fare',
                                                                                        'Average_fare', 'high_fare'])
        # Gathering tittles from name column and making categories based titles
        data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        data['Title'] = data['Title'].replace(
            ['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona', 'Countess', 'Lady', 'Sir'], 'Rare')
        data['Title'] = data['Title'].replace('Mlle', 'Miss')
        data['Title'] = data['Title'].replace('Ms', 'Miss')
        data['Title'] = data['Title'].replace('Mme', 'Mrs')

        # Creating age categories based on ranges of age
        data['Age_bin'] = pd.cut(data['Age'], bins=[0, 12, 20, 40, 120],
                                 labels=['Children', 'Teenage', 'Adult', 'Elder'])
        # Deleting unnecessary columns
        data = data.drop(['FamilySize', 'Name', 'Age', 'Fare'], axis=1)

        # Create Dummies (Binary)
        data = pd.get_dummies(data, columns=["Sex", "Age_bin", "Title", "Fare_bin", "Embarked"],
                              prefix=["Sex", "Age_type", "Title", "Fare_type", "Embarked_type"])

        return data
