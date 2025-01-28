import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys
import os

# Add the path to the scripts directory
sys.path.append(os.path.abspath('../scripts'))

from model import ModelTraining

class TestModelTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize the ModelTraining class with a sample data path
        cls.data_path = '../src/Data/rfms.csv'
        cls.model_training = ModelTraining(cls.data_path)
        
        # Create a sample dataframe for testing
        data = {
            'Amount': [100, 200, 150, 250],
            'Value': [50, 100, 75, 125],
            'total_transaction_amount': [300, 400, 350, 450],
            'average_transaction_amount': [75, 100, 87.5, 112.5],
            'transaction_count': [4, 4, 4, 4],
            'std_transaction_amount': [20, 30, 25, 35],
            'Label': ['Good', 'Bad', 'Good', 'Bad']
        }
        cls.model_training.df = pd.DataFrame(data)

    def test_split_data(self):
        self.model_training.split_data()
        self.assertEqual(self.model_training.X_train.shape[0], 3)
        self.assertEqual(self.model_training.X_test.shape[0], 1)
        self.assertEqual(self.model_training.y_train.shape[0], 3)
        self.assertEqual(self.model_training.y_test.shape[0], 1)

    def test_train_logistic_regression(self):
        self.model_training.split_data()
        self.model_training.train_logistic_regression()
        self.assertIsInstance(self.model_training.log_reg, LogisticRegression)
        self.assertEqual(len(self.model_training.y_pred_log_reg), self.model_training.X_test.shape[0])

    def test_train_random_forest(self):
        self.model_training.split_data()
        self.model_training.train_random_forest()
        self.assertIsInstance(self.model_training.rf, RandomForestClassifier)
        self.assertEqual(len(self.model_training.y_pred_rf), self.model_training.X_test.shape[0])

    def test_hyperparameter_tuning_log_reg(self):
        self.model_training.split_data()
        try:
            self.model_training.hyperparameter_tuning_log_reg()
        except Exception as e:
            self.fail(f"hyperparameter_tuning_log_reg() raised Exception unexpectedly: {e}")

    def test_hyperparameter_tuning_rf(self):
        self.model_training.split_data()
        try:
            self.model_training.hyperparameter_tuning_rf()
        except Exception as e:
            self.fail(f"hyperparameter_tuning_rf() raised Exception unexpectedly: {e}")

    def test_evaluate_models(self):
        self.model_training.split_data()
        self.model_training.train_logistic_regression()
        self.model_training.train_random_forest()
        try:
            self.model_training.evaluate_models()
        except Exception as e:
            self.fail(f"evaluate_models() raised Exception unexpectedly: {e}")

    def test_save_model(self):
        self.model_training.split_data()
        self.model_training.train_random_forest()
        try:
            self.model_training.save_model(self.model_training.rf, 'random_forest_model_test.pkl')
        except Exception as e:
            self.fail(f"save_model() raised Exception unexpectedly: {e}")

if __name__ == '__main__':
    unittest.main()
