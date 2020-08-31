import unittest
import pandas as pd
import os

from src.build_features import BuildFeatures


class TestBuildFeatures(unittest.TestCase):
    # Test set up
    def setUp(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.script_dir, os.pardir, 'data/test_data_build_features/test.csv')
        self.expected_path = os.path.join(self.script_dir, os.pardir,
                                          'data/test_data_build_features/expected_result.csv')

    # Testing number of columns
    def test_columns_length(self):
        # Exporting test data
        test_data = pd.read_csv(self.data_path)
        test_result = BuildFeatures().execute(test_data)

        # Exporting expected data
        expected_result = pd.read_csv(self.expected_path)

        self.assertEqual(len(test_result.columns), len(expected_result.columns))

    # Testing if columns were dropped
    def test_columns_removed(self):
        # Exporting test data
        test_data = pd.read_csv(self.data_path)
        test_result = BuildFeatures().execute(test_data)

        removed_columns = ['FamilySize', 'Name', 'Age', 'Fare']
        self.assertFalse(set(removed_columns).issubset(test_result.columns))

        # Testing if columns were dropped

    def test_columns_prefixes(self):
        # Exporting test data
        test_data = pd.read_csv(self.data_path)
        test_result = BuildFeatures().execute(test_data)

        # Embarked Prefixes
        added_embarked_columns_prefixes = ['Embarked_type_S', 'Embarked_type_Q', 'Embarked_type_C']
        self.assertTrue(set(added_embarked_columns_prefixes).issubset(test_result.columns))

        # Age Prefixes
        added_age_columns_prefixes = ['Age_type_Children', 'Age_type_Teenage', 'Age_type_Adult', 'Age_type_Elder']
        self.assertTrue(set(added_age_columns_prefixes).issubset(test_result.columns))

        # Title Prefixes
        added_title_columns_prefixes = ['Title_type_Master', 'Title_type_Miss',
                                        'Title_type_Mr', 'Title_type_Mrs', 'Title_type_Rare']
        self.assertTrue(set(added_title_columns_prefixes).issubset(test_result.columns))

        # Fare Prefixes
        added_fare_columns_prefixes = ['Fare_type_Low_fare', 'Fare_type_Median_fare',
                                       'Fare_type_Average_fare', 'Fare_type_High_fare']
        self.assertTrue(set(added_fare_columns_prefixes).issubset(test_result.columns))