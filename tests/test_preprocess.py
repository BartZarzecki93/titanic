import unittest
import pandas as pd
import os

from src.preprocess import Preprocessing


class TestPreprocess(unittest.TestCase):
    # Test set up
    def setUp(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.script_dir, os.pardir, 'data/test_data_preprocess/test.csv')
        self.expected_path = os.path.join(self.script_dir, os.pardir, 'data/test_data_preprocess/expected_result.csv')

    # Testing if length of columns and rows is correct
    def test_number_of_columns(self):

        # Exporting test data
        test_data = pd.read_csv(self.data_path, sep=";")
        test_result = Preprocessing().execute(test_data)

        # Exporting expected data

        expected_result = pd.read_csv(self.expected_path)

        self.assertEqual(len(test_result.columns), len(expected_result.columns))

    # Testing if columns were dropped in the preprocessing
    def test_columns_removed(self):

        # Exporting test data
        test_data = pd.read_csv(self.data_path, sep=";")
        test_result = Preprocessing().execute(test_data)

        removed_columns = ['Cabin', 'Ticket', 'PassengerId']
        self.assertFalse(set(removed_columns).issubset(test_result.columns))