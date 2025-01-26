import unittest
import pandas as pd
import os
from transaction_processor import TransactionProcessor  # Assuming the TransactionProcessor class is saved in a file named transaction_processor.py

class TestTransactionProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a sample DataFrame for testing
        data = {
            'TransactionId': ['T1', 'T2', 'T3'],
            'CustomerId': ['C1', 'C2', 'C3'],
            'Amount': [100.0, 200.0, 300.0],
            'TransactionStartTime': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'CurrencyCode': ['USD', 'EUR', 'JPY'],
            'ProviderId': ['P1', 'P2', 'P3'],
            'ProductId': ['PR1', 'PR2', 'PR3'],
            'ProductCategory': ['Cat1', 'Cat2', 'Cat3'],
            'ChannelId': ['Web', 'Android', 'iOS'],
            'PricingStrategy': [1, 2, 3],
            'Value': [100, 200, 300]
        }
        cls.test_df = pd.DataFrame(data)
        cls.processor = TransactionProcessor(cls.test_df)

    def test_aggregate_features(self):
        result_df = self.processor.aggregate_features()
        self.assertIn('total_transaction_amount', result_df.columns)
        self.assertIn('average_transaction_amount', result_df.columns)
        self.assertIn('transaction_count', result_df.columns)
        self.assertIn('std_transaction_amount', result_df.columns)

    def test_extract_time_features(self):
        result_df = self.processor.extract_time_features()
        self.assertIn('transaction_hour', result_df.columns)
        self.assertIn('transaction_day', result_df.columns)
        self.assertIn('transaction_month', result_df.columns)
        self.assertIn('transaction_year', result_df.columns)

    def test_encode_categorical_columns(self):
        result_df = self.processor.encode_categorical_columns()
        categorical_columns = ['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
        for col in categorical_columns:
            self.assertTrue(pd.api.types.is_integer_dtype(result_df[col]))

    def test_handle_missing_values(self):
        # Introduce missing values for testing
        self.processor.df.loc[0, 'Amount'] = None
        result_df = self.processor.handle_missing_values()
        self.assertFalse(result_df['Amount'].isnull().any())

    def test_normalize_or_standardize(self):
        result_df = self.processor.normalize_or_standardize(method='normalize')
        self.assertTrue((result_df[['Amount', 'Value', 'total_transaction_amount', 'average_transaction_amount', 'transaction_count', 'std_transaction_amount']] >= 0).all().all())
        self.assertTrue((result_df[['Amount', 'Value', 'total_transaction_amount', 'average_transaction_amount', 'transaction_count', 'std_transaction_amount']] <= 1).all().all())

        result_df = self.processor.normalize_or_standardize(method='standardize')
        self.assertAlmostEqual(result_df[['Amount', 'Value', 'total_transaction_amount', 'average_transaction_amount', 'transaction_count', 'std_transaction_amount']].mean().mean(), 0, places=1)
        self.assertAlmostEqual(result_df[['Amount', 'Value', 'total_transaction_amount', 'average_transaction_amount', 'transaction_count', 'std_transaction_amount']].std().mean(), 1, places=1)

    def test_save_final_dataframe(self):
        self.processor.save_final_dataframe(filename='test_final.csv')
        self.assertTrue(os.path.exists('test_final.csv'))
        os.remove('test_final.csv')

if __name__ == '__main__':
    unittest.main()