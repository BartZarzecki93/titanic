import pandas as pd


class BuildFeatures:
    def execute(self, data):
        # Creating sex categories based on sex column
        dummies_sex_columns = ['female', 'male']
        data['Sex'] = data['Sex'].map({'female': 1, 'male': 2}).astype(int)
        data['Sex_bin'] = pd.cut(data['Sex'], bins=[0, 1.5, 2.5], labels=dummies_sex_columns)

        # Creating embarked categories based on embarked column
        dummies_sex_columns = ['S', 'Q', 'C']
        data['Embarked'] = data['Embarked'].map({'S': 1, 'Q': 2, 'C': 3}).astype(int)
        data['Embarked_bin'] = pd.cut(data['Embarked'], bins=[0, 1.5, 2.5, 3.5], labels=dummies_sex_columns)

        # Building family size column based on SibSp and Parch
        data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

        # Is Alone? questioning if person was alone or no, data retrieved from FamilySize
        data["IsAlone"] = 0
        data.loc[data["FamilySize"] == 1, "IsAlone"] = 1

        # Creating fare categories based on ranges of prices
        dummies_age_columns = ['Low_fare', 'Median_fare', 'Average_fare', 'High_fare']
        data['Fare_bin'] = pd.cut(data['Fare'], bins=[0, 7.91, 14.45, 31, 120], labels=dummies_age_columns)

        # Gathering tittles from name column and making categories based on titles
        data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        data['Title'] = data['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        data['Title'] = data['Title'].replace('Mlle', 'Miss')
        data['Title'] = data['Title'].replace('Ms', 'Miss')
        data['Title'] = data['Title'].replace('Mme', 'Mrs')
        title_mapping = {'Rare': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Mr': 5}
        data['Title'] = data['Title'].map(title_mapping).astype(int)

        dummies_title_columns = ['Rare', 'Miss', 'Mrs', 'Master', 'Mr']
        data['Title_bin'] = pd.cut(data['Title'], bins=[0, 1.5, 2.5, 3.5, 4.5, 5.5], labels=dummies_title_columns)

        # Creating age categories based on ranges of age
        data['Age_bin'] = pd.cut(data['Age'], bins=[0, 12, 20, 40, 120],
                                 labels=['Children', 'Teenage', 'Adult', 'Elder'])

        # Deleting unnecessary columns
        data = data.drop(['FamilySize', 'Name', 'Age', 'Fare', 'Title', 'Sex', 'Embarked'], axis=1)

        # Create Dummies (Binary)
        data = pd.get_dummies(data, columns=["Sex_bin", "Age_bin", "Title_bin", "Fare_bin", "Embarked_bin"],
                              prefix=["Sex_type", "Age_type", "Title_type", "Fare_type", "Embarked_type"])

        return data
