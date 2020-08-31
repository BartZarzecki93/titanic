import unittest
import os
import pandas as pd

from src.predict import Predict


class TestPredict(unittest.TestCase):
    # Test set up
    def setUp(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.script_dir, os.pardir, 'data/test_data_predict/test.csv')
        self.model_data_path = os.path.join(self.script_dir, os.pardir, 'data/test_data_predict/model.pkl')
        self.expected_path = os.path.join(self.script_dir, os.pardir, 'data/test_data_predict/expected_result.csv')

    # Testing if columns were dropped
    def test_columns_removed(self):
        # Exporting test data
        test_data = pd.read_csv(self.data_path, sep=";")
        test_result = Predict().execute(test_data, self.model_data_path)

        removed_columns = ['FamilySize', 'Name', 'Age', 'Fare', 'Title', 'Sex', 'Embarked', 'PassengerId']
        self.assertFalse(set(removed_columns).issubset(test_result.columns))

    # Testing if columns were added
    def test_column_added(self):
        # Exporting test data
        test_data = pd.read_csv(self.data_path, sep=";")
        test_result = Predict().execute(test_data, self.model_data_path)

        added_columns = ['Prediction', 'Target']
        self.assertTrue(set(added_columns).issubset(test_result.columns))

    # Testing length of columns
    def test_number_of_columns(self):
        # Exporting test data
        test_data = pd.read_csv(self.data_path, sep=";")
        test_result = Predict().execute(test_data, self.model_data_path)

        self.assertEqual(len(test_result.columns), 24)
