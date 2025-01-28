import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

class FeatureEngineering:
    def __init__(self, data_path):
        try:
            self.df = pd.read_csv(data_path)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def create_aggregate_features(self):
        try:
            agg_features = self.df.groupby('CustomerId').agg(
                total_transaction_amount=('Amount', 'sum'),
                average_transaction_amount=('Amount', 'mean'),
                transaction_count=('TransactionId', 'count'),
                std_transaction_amount=('Amount', 'std')
            ).reset_index()

            # Merge aggregate features back into the original dataframe
            self.df = pd.merge(self.df, agg_features, on='CustomerId', how='left')
            print("Aggregate features created successfully.")
        except Exception as e:
            print(f"Error creating aggregate features: {e}")
        return self.df

    def extract_time_features(self):
        try:
            self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
            self.df['transaction_hour'] = self.df['TransactionStartTime'].dt.hour
            self.df['transaction_day'] = self.df['TransactionStartTime'].dt.day
            self.df['transaction_month'] = self.df['TransactionStartTime'].dt.month
            self.df['transaction_year'] = self.df['TransactionStartTime'].dt.year
            print("Time features extracted successfully.")
        except Exception as e:
            print(f"Error extracting time features: {e}")
        return self.df

    def label_encode(self, columns):
        try:
            label_encoder = LabelEncoder()
            for col in columns:
                self.df[col] = label_encoder.fit_transform(self.df[col])
            print("Label encoding completed successfully.")
        except Exception as e:
            print(f"Error in label encoding: {e}")
        return self.df

    def check_missing_values(self):
        try:
            missing_values = self.df.isnull().sum()
            print("### Missing Values ###")
            print(missing_values)
        except Exception as e:
            print(f"Error checking missing values: {e}")

    def handle_missing_values(self):
        try:
            self.df.dropna(subset=['std_transaction_amount'], inplace=True)
            print("Missing values handled successfully.")
        except Exception as e:
            print(f"Error handling missing values: {e}")
        return self.df

    def scale_numerical_features(self, method='normalize'):
        try:
            numerical_columns = ['Amount', 'Value', 'total_transaction_amount', 'average_transaction_amount', 'transaction_count', 'std_transaction_amount']
            
            if method == 'normalize':
                scaler = MinMaxScaler()
            elif method == 'standardize':
                scaler = StandardScaler()
            else:
                raise ValueError("Method must be 'normalize' or 'standardize'")
            
            self.df[numerical_columns] = scaler.fit_transform(self.df[numerical_columns])
            print(f"Numerical features scaled using {method} method.")
        except Exception as e:
            print(f"Error scaling numerical features: {e}")
        return self.df

    def save_cleaned_data(self, output_path):
        try:
            self.df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to {output_path}")
        except Exception as e:
            print(f"Error saving cleaned data: {e}")