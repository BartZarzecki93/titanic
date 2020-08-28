import unittest

from src.predict import Predict


class TestPredict(unittest.TestCase):
    # Testing if columns were dropped
    def test_columns_removed(self):
        # Exporting test data
        test_data = "../data/test_data_predict/test.csv"
        model_path = '../data/model.pkl'
        test_result = Predict().execute(test_data, model_path)

        removed_columns = ['FamilySize', 'Name', 'Age', 'Fare', 'Cabin', 'Ticket', 'PassengerId']
        self.assertFalse(set(removed_columns).issubset(test_result.columns))

    # Testing if columns were added
    def test_column_added(self):
        # Exporting test data
        test_data = "../data/test_data_predict/test.csv"
        model_path = '../data/model.pkl'
        test_result = Predict().execute(test_data, model_path)

        added_columns = ['Prediction', 'Target']
        self.assertTrue(set(added_columns).issubset(test_result.columns))

    # Testing length of columns
    def test_columns_length(self):
        # Exporting test data
        test_data = "../data/test_data_predict/test.csv"
        model_path = '../data/model.pkl'
        test_result = Predict().execute(test_data, model_path)

        self.assertEqual(len(test_result.columns), 24)
