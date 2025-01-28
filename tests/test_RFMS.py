import unittest
import pandas as pd
import sys
import os

# Add the path to the scripts directory
sys.path.append(os.path.abspath('../scripts'))

from RFMS import RFMS

class TestRFMS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize the RFMS class with a sample data path
        cls.data_path = '../src/Data/featured.csv'
        cls.rfms = RFMS(cls.data_path)
        
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
        cls.rfms.df = pd.DataFrame(data)

    def test_calculate_rfms_features(self):
        df = self.rfms.calculate_rfms_features()
        self.assertIn('Recency', df.columns)
        self.assertIn('Frequency', df.columns)
        self.assertIn('Monetary', df.columns)
        self.assertIn('Score', df.columns)

    def test_check_distribution(self):
        try:
            self.rfms.check_distribution()
        except Exception as e:
            self.fail(f"check_distribution() raised Exception unexpectedly: {e}")

    def test_visualize_rfms(self):
        try:
            self.rfms.visualize_rfms()
        except Exception as e:
            self.fail(f"visualize_rfms() raised Exception unexpectedly: {e}")

    def test_assign_labels(self):
        df = self.rfms.assign_labels(threshold=0.5)
        self.assertIn('Label', df.columns)
        self.assertTrue(df['Label'].isin(['Good', 'Bad']).all())

    def test_plot_labels(self):
        try:
            self.rfms.plot_labels()
        except Exception as e:
            self.fail(f"plot_labels() raised Exception unexpectedly: {e}")

    def test_save_rfms_data(self):
        output_path = '../src/Data/rfms_test.csv'
        try:
            self.rfms.save_rfms_data(output_path)
        except Exception as e:
            self.fail(f"save_rfms_data() raised Exception unexpectedly: {e}")

if __name__ == '__main__':
    unittest.main()