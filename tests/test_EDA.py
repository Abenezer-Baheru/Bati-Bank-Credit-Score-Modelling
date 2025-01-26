import unittest
import pandas as pd
import os
from eda import EDA  # Assuming the EDA class is saved in a file named eda.py

class TestEDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a sample DataFrame for testing
        data = {
            'TransactionId': ['T1', 'T2', 'T3'],
            'BatchId': ['B1', 'B2', 'B3'],
            'AccountId': ['A1', 'A2', 'A3'],
            'SubscriptionId': ['S1', 'S2', 'S3'],
            'CustomerId': ['C1', 'C2', 'C3'],
            'CurrencyCode': ['USD', 'EUR', 'JPY'],
            'CountryCode': [1, 2, 3],
            'ProviderId': ['P1', 'P2', 'P3'],
            'ProductId': ['PR1', 'PR2', 'PR3'],
            'ProductCategory': ['Cat1', 'Cat2', 'Cat3'],
            'ChannelId': ['Web', 'Android', 'iOS'],
            'Amount': [100.0, 200.0, 300.0],
            'Value': [100, 200, 300],
            'TransactionStartTime': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'PricingStrategy': [1, 2, 3],
            'FraudResult': [0, 1, 0]
        }
        cls.test_df = pd.DataFrame(data)
        cls.test_file_path = 'test_data.csv'
        cls.test_df.to_csv(cls.test_file_path, index=False)
        cls.eda = EDA(cls.test_file_path)

    @classmethod
    def tearDownClass(cls):
        # Clean up the test file
        os.remove(cls.test_file_path)

    def test_overview_of_data(self):
        shape, dtypes, missing_values = self.eda.overview_of_data()
        self.assertEqual(shape, (3, 16))
        self.assertTrue(isinstance(dtypes, pd.Series))
        self.assertTrue(isinstance(missing_values, pd.Series))

    def test_summary_statistics(self):
        numerical_summary, categorical_summary = self.eda.summary_statistics()
        self.assertTrue(isinstance(numerical_summary, pd.DataFrame))
        self.assertTrue(isinstance(categorical_summary, pd.DataFrame))

    def test_distribution_of_numerical_features(self):
        # This method generates plots, so we just check if it runs without errors
        try:
            self.eda.distribution_of_numerical_features()
        except Exception as e:
            self.fail(f"distribution_of_numerical_features() raised {e}")

    def test_distribution_of_categorical_features(self):
        # This method generates plots, so we just check if it runs without errors
        try:
            self.eda.distribution_of_categorical_features()
        except Exception as e:
            self.fail(f"distribution_of_categorical_features() raised {e}")

    def test_correlation_analysis(self):
        correlation_matrix, means, stds, counts = self.eda.correlation_analysis()
        self.assertTrue(isinstance(correlation_matrix, pd.DataFrame))
        self.assertTrue(isinstance(means, pd.Series))
        self.assertTrue(isinstance(stds, pd.Series))
        self.assertTrue(isinstance(counts, pd.Series))

    def test_identify_missing_values(self):
        # This method generates plots, so we just check if it runs without errors
        try:
            self.eda.identify_missing_values()
        except Exception as e:
            self.fail(f"identify_missing_values() raised {e}")

    def test_detect_outliers(self):
        outliers_dict = self.eda.detect_outliers()
        self.assertTrue(isinstance(outliers_dict, dict))

    def test_handle_outliers(self):
        cleaned_data = self.eda.handle_outliers()
        self.assertTrue(isinstance(cleaned_data, pd.DataFrame))

if __name__ == '__main__':
    unittest.main()