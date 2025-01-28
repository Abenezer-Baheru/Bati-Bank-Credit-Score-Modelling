import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, df):
        self.df = df

    def overview_of_data(self):
        try:
            print("### Overview of Data ###")
            print(f"Number of Rows: {self.df.shape[0]}")
            print(f"Number of Columns: {self.df.shape[1]}")
            print("\nData Types:\n", self.df.dtypes)
            print("\nMissing Values:\n", self.df.isnull().sum())
            return self.df.shape, self.df.dtypes, self.df.isnull().sum()
        except Exception as e:
            print(f"An error occurred: {e}")

    def summary_statistics(self):
        try:
            numerical_summary = self.df.describe(include=['float64', 'int64'])
            categorical_summary = self.df.describe(include=['object'])

            print("### Summary Statistics for Numerical Data ###")
            print(numerical_summary)

            print("\n### Summary Statistics for Categorical Data ###")
            print(categorical_summary)

            return numerical_summary, categorical_summary
        except Exception as e:
            print(f"An error occurred: {e}")

    def count_unique_values(self):
        try:
            unique_counts = {col: self.df[col].nunique() for col in self.df.columns}
            for col, count in unique_counts.items():
                print(f"Unique values in {col}: {count}")
            return unique_counts
        except Exception as e:
            print(f"An error occurred: {e}")

    def find_unique_values(self, columns):
        try:
            unique_values = {col: self.df[col].unique() for col in columns}
            for col, values in unique_values.items():
                print(f"Unique values in {col}: {values}\n")
            return unique_values
        except Exception as e:
            print(f"An error occurred: {e}")

    def plot_numerical_features(self):
        try:
            numerical_features = ['Amount', 'Value']
            categorical_like_numerical_features = ['CountryCode', 'PricingStrategy', 'FraudResult']

            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

            for i, feature in enumerate(numerical_features):
                row = i // 3
                col = i % 3
                self.df[feature].plot(kind='hist', bins=50, color='skyblue', edgecolor='black', ax=axs[row, col])
                unique_count = self.df[feature].nunique()
                axs[row, col].set_title(f'Distribution of {feature} (Unique Values: {unique_count})')
                axs[row, col].set_xlabel(feature)
                axs[row, col].set_ylabel('Frequency')
                axs[row, col].grid(True)

            for i, feature in enumerate(categorical_like_numerical_features, len(numerical_features)):
                row = i // 3
                col = i % 3
                sns.countplot(data=self.df, x=feature, palette='pastel', ax=axs[row, col])
                unique_count = self.df[feature].nunique()
                axs[row, col].set_title(f'Distribution of {feature} (Unique Values: {unique_count})')
                axs[row, col].set_xlabel(feature)
                axs[row, col].set_ylabel('Count')
                axs[row, col].tick_params(axis='x', rotation=45)
                axs[row, col].grid(True)

            for i in range(len(numerical_features) + len(categorical_like_numerical_features), 6):
                fig.delaxes(axs.flatten()[i])

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"An error occurred: {e}")

    def plot_categorical_features(self):
        try:
            categorical_features = ['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']

            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

            for i, feature in enumerate(categorical_features):
                row = i // 3
                col = i % 3
                sns.countplot(data=self.df, x=feature, palette='pastel', ax=axs[row, col])
                unique_count = self.df[feature].nunique()
                axs[row, col].set_title(f'Distribution of {feature} (Unique values: {unique_count})')
                axs[row, col].set_xlabel(feature)
                axs[row, col].set_ylabel('Count')
                axs[row, col].tick_params(axis='x', rotation=45)
                axs[row, col].grid(True)

            for i in range(len(categorical_features), 6):
                fig.delaxes(axs.flatten()[i])

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"An error occurred: {e}")

    def correlation_analysis(self):
        try:
            numeric_data = self.df.select_dtypes(include=['float64', 'int64'])
            correlation_matrix = numeric_data.corr()
            print("### Correlation Analysis ###")
            print("\nCorrelation Matrix:")
            print(correlation_matrix)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='Blues', square=True, cbar_kws={"shrink": .8}, linewidths=.5, linecolor='black')
            plt.title('Correlation Heatmap', fontsize=16)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()
            
            return correlation_matrix
        except Exception as e:
            print(f"An error occurred: {e}")

    def identify_missing_values(self):
        try:
            missing_values = self.df.isnull().sum()
            missing_values = missing_values[missing_values > 0]
            print("### Missing Values ###")
            print(missing_values)
            return missing_values
        except Exception as e:
            print(f"An error occurred: {e}")

    def detect_outliers_with_visualization(self):
        try:
            numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
            outliers_dict = {}
            
            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), squeeze=False)
            axs = axs.flatten()

            for idx, col in enumerate(numerical_cols):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outliers_dict[col] = len(outliers)
                
                print(f"{len(outliers)} outliers detected in {col} using IQR method.")
                
                sns.boxplot(data=self.df, y=col, ax=axs[idx])
                axs[idx].set_title(f'Box Plot of {col}')
                axs[idx].set_ylabel(col)
                axs[idx].grid(True)

            for i in range(len(numerical_cols), len(axs)):
                fig.delaxes(axs[i])

            plt.tight_layout()
            plt.show()
            
            return outliers_dict
        except Exception as e:
            print(f"An error occurred: {e}")