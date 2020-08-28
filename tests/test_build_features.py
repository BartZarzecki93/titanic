import unittest
import pandas as pd
from src.build_features import BuildFeatures


class TestBuildFeatures(unittest.TestCase):

    # Testing number of rows and columns after build feature function
    def test_rows_columns_length(self):

        # Exporting test data
        test_data = pd.read_csv('../data/test_data_build_features/test.csv')
        test_result = BuildFeatures().execute(test_data)

        # Exporting expected data
        expected_result = pd.read_csv('../data/test_data_build_features/expected_result.csv')

        self.assertEqual(len(test_result.columns), len(expected_result.columns))
        self.assertEqual(test_result.shape[0], expected_result.shape[0])

    # Testing if columns were dropped in the build features
    def test_columns_removed(self):
        test_data = pd.read_csv('../data/test_data_build_features/test.csv')
        result = BuildFeatures().execute(test_data)

        self.assertFalse('FamilySize' in result)
        self.assertFalse('Name' in result)
        self.assertFalse('Age' in result)
        self.assertFalse('Fare' in result)
