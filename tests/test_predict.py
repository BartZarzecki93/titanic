import unittest

from src.predict import Predict


class TestPredict(unittest.TestCase):
    # Testing if columns were dropped in the preprocess
    def test_columns_removed_preprocess(self):
        # Exporting test data
        test_data = "../data/test_data_predict/test.csv"
        model_path = '../data/model.pkl'

        test_result = Predict().execute(test_data, model_path)

        self.assertEqual('Cabin' in test_result, False)
        self.assertEqual('Ticket' in test_result, False)
        self.assertEqual('PassengerId' in test_result, False)

    # Testing if columns were dropped in the build features
    def test_columns_removed_build_features(self):
        # Exporting test data
        test_data = "../data/test_data_predict/test.csv"
        model_path = '../data/model.pkl'

        test_result = Predict().execute(test_data, model_path)

        self.assertFalse('FamilySize' in test_result)
        self.assertFalse('Name' in test_result)
        self.assertFalse('Age' in test_result)
        self.assertFalse('Fare' in test_result)

    def test_columns_length(self):
        # Exporting test data
        test_data = "../data/test_data_predict/test.csv"
        model_path = '../data/model.pkl'

        test_result = Predict().execute(test_data, model_path)

        self.assertEqual(len(test_result.columns), 24)
