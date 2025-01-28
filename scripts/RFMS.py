import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class RFMS:
    def __init__(self, data_path):
        try:
            self.df = pd.read_csv(data_path)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def calculate_rfms_features(self):
        try:
            # Convert TransactionStartTime to datetime
            self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
            
            # Calculate Recency: Days since the last transaction
            recency_df = self.df.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
            recency_df['Recency'] = (self.df['TransactionStartTime'].max() - recency_df['TransactionStartTime']).dt.days
            
            # Calculate Frequency: Number of transactions per customer
            frequency_df = self.df.groupby('CustomerId')['TransactionId'].count().reset_index()
            frequency_df.columns = ['CustomerId', 'Frequency']
            
            # Calculate Monetary: Total transaction amount per customer
            monetary_df = self.df.groupby('CustomerId')['Amount'].sum().reset_index()
            monetary_df.columns = ['CustomerId', 'Monetary']
            
            # Merge RFMS features into the original dataframe
            rfms_df = pd.merge(recency_df, frequency_df, on='CustomerId')
            rfms_df = pd.merge(rfms_df, monetary_df, on='CustomerId')
            
            # Calculate Score: A simple combination of Recency, Frequency, and Monetary
            rfms_df['Score'] = rfms_df['Frequency'] * rfms_df['Monetary'] / (rfms_df['Recency'] + 1)
            
            # Merge RFMS features back into the original dataframe
            self.df = pd.merge(self.df, rfms_df, on='CustomerId', how='left')
            print("RFMS features calculated successfully.")
        except Exception as e:
            print(f"Error calculating RFMS features: {e}")
        return self.df

    def check_distribution(self):
        try:
            numerical_columns = ['Recency', 'Frequency', 'Monetary', 'Score']
            plt.figure(figsize=(12, 8))
            
            for i, col in enumerate(numerical_columns, 1):
                plt.subplot(2, 2, i)
                sns.histplot(self.df[col], kde=True)
                plt.title(f'Distribution of {col}')
            
            plt.tight_layout()
            plt.show()
            print("Distribution check completed successfully.")
        except Exception as e:
            print(f"Error checking distribution: {e}")

    def visualize_rfms(self):
        try:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=self.df, x='Recency', y='Frequency', hue='Score', palette='coolwarm')
            plt.title('RFMS Space')
            plt.xlabel('Recency')
            plt.ylabel('Frequency')
            plt.legend(title='Score')
            plt.grid(True)
            plt.show()
            print("RFMS visualization completed successfully.")
        except Exception as e:
            print(f"Error visualizing RFMS: {e}")

    def assign_labels(self, threshold=0.5):
        try:
            self.df['Label'] = self.df['Score'].apply(lambda x: 'Good' if x >= threshold else 'Bad')
            print("Labels assigned successfully.")
        except Exception as e:
            print(f"Error assigning labels: {e}")
        return self.df

    def plot_labels(self):
        try:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=self.df, x='Label', palette='coolwarm')
            plt.title('Number of Customers by Label')
            plt.xlabel('Label')
            plt.ylabel('Number of Customers')
            plt.grid(True)
            plt.show()
            print("Label plot completed successfully.")
        except Exception as e:
            print(f"Error plotting labels: {e}")

    def save_rfms_data(self, output_path):
        try:
            self.df.to_csv(output_path, index=False)
            print(f"RFMS data saved to {output_path}")
        except Exception as e:
            print(f"Error saving RFMS data: {e}")