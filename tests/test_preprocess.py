import unittest
import pandas as pd
from src.preprocess import Preprocessing


class TestPreprocess(unittest.TestCase):

    # Testing if length of columns and rows is correct
    def test_number_of_rows_columns_(self):

        # Exporting test data
        test_data = pd.read_csv('../data/test_data_preprocess/test.csv', sep=";")
        test_result = Preprocessing().execute(test_data)

        # Exporting expected data
        expected_result = pd.read_csv('../data/test_data_preprocess/expected_result.csv')

        self.assertEqual(len(test_result.columns), len(expected_result.columns))
        self.assertEqual(test_result.shape[0], expected_result.shape[0])

    # Testing if columns were dropped in the preprocessing
    def test_columns_removed_preprocess(self):

        # Exporting test data
        test_data = pd.read_csv('../data/test_data_preprocess/test.csv', sep=";")
        test_result = Preprocessing().execute(test_data)

        self.assertEqual('Cabin' in test_result, False)
        self.assertEqual('Ticket' in test_result, False)
        self.assertEqual('PassengerId' in test_result, False)
