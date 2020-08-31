import unittest
import pandas as pd
import pickle as pkl
import os

from src.train import Train

class TestTrain(unittest.TestCase):

    # Test set up
    def setUp(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.script_dir, os.pardir, 'data/test_data_train/test.csv')
        self.model_data_path = os.path.join(self.script_dir, os.pardir, 'data/test_data_train/model.pkl')
        self.expected_model_path = os.path.join(self.script_dir, os.pardir,
                                                'data/test_data_train/expected_model.pkl')

    # Testing if column was deleted from training data
    def test_column_removed(self):
        # Exporting test data
        test_data = pd.read_csv(self.data_path)
        test_data_and_model = Train().execute(test_data, self.model_data_path)
        self.assertEqual('Survived' in test_data_and_model[0], False)

    # Testing types of models
    def test_model(self):
        # Exporting test data and model
        test_data = pd.read_csv(self.data_path)
        test_data_and_model = Train().execute(test_data, self.model_data_path)

        # Unpickling files
        model_unpickle = open(self.expected_model_path, 'rb')
        model = pkl.load(model_unpickle)
        model_unpickle.close()

        print(test_data_and_model[1])
        print(model)

        self.assertEqual(type((test_data_and_model[1])), type(model))
