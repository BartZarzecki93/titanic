import unittest
import pandas as pd
import pickle as pkl

from src.train import Train

class TestTrain(unittest.TestCase):

    # Testing if column was deleted from training data
    def test_column_removed(self):
        # Exporting test data
        test_data = pd.read_csv("../data/test_data_train/test.csv")
        model_path = '../data/model.pkl'
        test_data_and_model = Train().execute(test_data, model_path)

        self.assertEqual('Survived' in test_data_and_model[0], False)

    # Testing types of models
    def test_model(self):
        # Exporting test data and model
        test_data = pd.read_csv("../data/test_data_train/test.csv")
        model_path = '../data/model.pkl'
        test_data_and_model = Train().execute(test_data, model_path)

        # Unpickling files
        expected_model ="../data/test_data_train/expected_model.pkl"
        model_unpickle = open(expected_model, 'rb')
        model = pkl.load(model_unpickle)
        model_unpickle.close()

        print(test_data_and_model[1])
        print(model)

        self.assertEqual(type((test_data_and_model[1])), type(model))
