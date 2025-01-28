import sys
import os
import unittest
import pandas as pd

# Add the path to the scripts directory
sys.path.append(os.path.abspath('../scripts'))

# Import the EDA class from EDA moudle
from EDA import EDA

class TestEDA(unittest.TestCase):
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
            'Value': [100, 200, 300],
            'CountryCode': [1, 2, 3],
            'FraudResult': [0, 1, 0]
        }
        cls.test_df = pd.DataFrame(data)
        cls.eda = EDA(cls.test_df)

    def test_overview_of_data(self):
        shape, dtypes, missing_values = self.eda.overview_of_data()
        self.assertEqual(shape, (3, 13))
        self.assertTrue(isinstance(dtypes, pd.Series))
        self.assertTrue(isinstance(missing_values, pd.Series))

    def test_summary_statistics(self):
        numerical_summary, categorical_summary = self.eda.summary_statistics()
        self.assertTrue(isinstance(numerical_summary, pd.DataFrame))
        self.assertTrue(isinstance(categorical_summary, pd.DataFrame))

    def test_count_unique_values(self):
        unique_counts = self.eda.count_unique_values()
        self.assertTrue(isinstance(unique_counts, dict))
        self.assertEqual(unique_counts['TransactionId'], 3)

    def test_find_unique_values(self):
        columns = ['CurrencyCode', 'CountryCode', 'ChannelId', 'ProviderId', 'PricingStrategy', 'FraudResult', 'ProductId', 'ProductCategory']
        unique_values = self.eda.find_unique_values(columns)
        self.assertTrue(isinstance(unique_values, dict))
        self.assertEqual(len(unique_values['CurrencyCode']), 3)

    def test_plot_numerical_features(self):
        # This method generates plots, so we just check if it runs without errors
        try:
            self.eda.plot_numerical_features()
        except Exception as e:
            self.fail(f"plot_numerical_features() raised {e}")

    def test_plot_categorical_features(self):
        # This method generates plots, so we just check if it runs without errors
        try:
            self.eda.plot_categorical_features()
        except Exception as e:
            self.fail(f"plot_categorical_features() raised {e}")

    def test_correlation_analysis(self):
        correlation_matrix = self.eda.correlation_analysis()
        self.assertTrue(isinstance(correlation_matrix, pd.DataFrame))

    def test_identify_missing_values(self):
        missing_values = self.eda.identify_missing_values()
        self.assertTrue(isinstance(missing_values, pd.Series))

    def test_detect_outliers_with_visualization(self):
        outliers_dict = self.eda.detect_outliers_with_visualization()
        self.assertTrue(isinstance(outliers_dict, dict))

if __name__ == '__main__':
    unittest.main()