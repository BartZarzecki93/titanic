import unittest
import pandas as pd

from src.train import Train


class TestTrain(unittest.TestCase):

    # Testing if column was deleted from training data
    def test_column_removed(self):
        test_data = pd.read_csv("../data/test_data_train/test.csv")
        model_path = '../data/model.pkl'

        test_result = Train().execute(test_data, model_path)

        self.assertEqual('Survived' in test_result, False)
