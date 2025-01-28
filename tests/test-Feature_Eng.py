import unittest
import pandas as pd
import sys
import os

# Add the path to the scripts directory
sys.path.append(os.path.abspath('../scripts'))

from Feature_Eng import FeatureEngineering

class TestFeatureEngineering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize the FeatureEngineering class with a sample data path
        cls.data_path = '../src/Data/data.csv'
        cls.feature_eng = FeatureEngineering(cls.data_path)
        
        # Create a sample dataframe for testing
        data = {
            'CustomerId': [1, 2, 1, 2],
            'TransactionId': [101, 102, 103, 104],
            'Amount': [100, 200, 150, 250],
            'TransactionStartTime': ['2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-01 12:00:00', '2023-01-02 13:00:00'],
            'CurrencyCode': ['USD', 'EUR', 'USD', 'EUR'],
            'ProviderId': ['P1', 'P2', 'P1', 'P2'],
            'ProductId': ['Prod1', 'Prod2', 'Prod1', 'Prod2'],
            'ProductCategory': ['Cat1', 'Cat2', 'Cat1', 'Cat2'],
            'ChannelId': ['C1', 'C2', 'C1', 'C2'],
            'PricingStrategy': ['Strategy1', 'Strategy2', 'Strategy1', 'Strategy2'],
            'FraudResult': ['No', 'Yes', 'No', 'Yes']
        }
        cls.feature_eng.df = pd.DataFrame(data)

    def test_create_aggregate_features(self):
        df = self.feature_eng.create_aggregate_features()
        self.assertIn('total_transaction_amount', df.columns)
        self.assertIn('average_transaction_amount', df.columns)
        self.assertIn('transaction_count', df.columns)
        self.assertIn('std_transaction_amount', df.columns)

    def test_extract_time_features(self):
        df = self.feature_eng.extract_time_features()
        self.assertIn('transaction_hour', df.columns)
        self.assertIn('transaction_day', df.columns)
        self.assertIn('transaction_month', df.columns)
        self.assertIn('transaction_year', df.columns)

    def test_label_encode(self):
        categorical_columns = ['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
        df = self.feature_eng.label_encode(categorical_columns)
        df = self.feature_eng.label_encode(['FraudResult'])
        for col in categorical_columns + ['FraudResult']:
            self.assertTrue(pd.api.types.is_integer_dtype(df[col]))

    def test_check_missing_values(self):
        try:
            self.feature_eng.check_missing_values()
        except Exception as e:
            self.fail(f"check_missing_values() raised Exception unexpectedly: {e}")

    def test_handle_missing_values(self):
        df = self.feature_eng.handle_missing_values()
        self.assertFalse(df['std_transaction_amount'].isnull().any())

    def test_scale_numerical_features(self):
        df = self.feature_eng.scale_numerical_features(method='normalize')
        numerical_columns = ['Amount', 'Value', 'total_transaction_amount', 'average_transaction_amount', 'transaction_count', 'std_transaction_amount']
        for col in numerical_columns:
            self.assertTrue((df[col] >= 0).all() and (df[col] <= 1).all())

    def test_save_cleaned_data(self):
        output_path = '../src/data/featured_test.csv'
        try:
            self.feature_eng.save_cleaned_data(output_path)
        except Exception as e:
            self.fail(f"save_cleaned_data() raised Exception unexpectedly: {e}")

if __name__ == '__main__':
    unittest.main()