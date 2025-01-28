import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

class ModelTraining:
    def __init__(self, data_path):
        try:
            self.df = pd.read_csv(data_path)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def split_data(self):
        try:
            # Define features and target variable
            self.X = self.df[['Amount', 'Value', 'total_transaction_amount', 'average_transaction_amount', 'transaction_count', 'std_transaction_amount']]
            self.y = self.df['Label']

            # Split the data into training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

            # Display the shapes of the training and testing sets
            print(f"X_train shape: {self.X_train.shape}")
            print(f"X_test shape: {self.X_test.shape}")
            print(f"y_train shape: {self.y_train.shape}")
            print(f"y_test shape: {self.y_test.shape}")
        except Exception as e:
            print(f"Error splitting data: {e}")

    def train_logistic_regression(self):
        try:
            # Initialize and train the Logistic Regression model
            self.log_reg = LogisticRegression(random_state=42)
            self.log_reg.fit(self.X_train, self.y_train)

            # Predict on the test set
            self.y_pred_log_reg = self.log_reg.predict(self.X_test)
            print("Logistic Regression model trained successfully.")
        except Exception as e:
            print(f"Error training Logistic Regression model: {e}")

    def train_random_forest(self):
        try:
            # Initialize and train the Random Forest model
            self.rf = RandomForestClassifier(random_state=42)
            self.rf.fit(self.X_train, self.y_train)

            # Predict on the test set
            self.y_pred_rf = self.rf.predict(self.X_test)
            print("Random Forest model trained successfully.")
        except Exception as e:
            print(f"Error training Random Forest model: {e}")

    def hyperparameter_tuning_log_reg(self):
        try:
            # Define the parameter grid for Logistic Regression
            param_grid_log_reg = {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }

            # Initialize Grid Search
            grid_search_log_reg = GridSearchCV(LogisticRegression(random_state=42), param_grid_log_reg, cv=5, scoring='accuracy')
            grid_search_log_reg.fit(self.X_train, self.y_train)

            # Best parameters and score
            print(f"Best parameters for Logistic Regression: {grid_search_log_reg.best_params_}")
            print(f"Best score for Logistic Regression: {grid_search_log_reg.best_score_}")
        except Exception as e:
            print(f"Error in hyperparameter tuning for Logistic Regression: {e}")

    def hyperparameter_tuning_rf(self):
        try:
            # Define the parameter grid for Random Forest
            param_grid_rf = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            # Initialize Random Search
            random_search_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, n_iter=10, cv=5, scoring='accuracy', random_state=42)
            random_search_rf.fit(self.X_train, self.y_train)

            # Best parameters and score
            print(f"Best parameters for Random Forest: {random_search_rf.best_params_}")
            print(f"Best score for Random Forest: {random_search_rf.best_score_}")
        except Exception as e:
            print(f"Error in hyperparameter tuning for Random Forest: {e}")

    def evaluate_models(self):
        try:
            # Evaluate Logistic Regression
            accuracy_log_reg = accuracy_score(self.y_test, self.y_pred_log_reg)
            precision_log_reg = precision_score(self.y_test, self.y_pred_log_reg, pos_label='Good')
            recall_log_reg = recall_score(self.y_test, self.y_pred_log_reg, pos_label='Good')
            f1_log_reg = f1_score(self.y_test, self.y_pred_log_reg, pos_label='Good')
            roc_auc_log_reg = roc_auc_score(self.y_test, self.log_reg.predict_proba(self.X_test)[:, 1])

            # Evaluate Random Forest
            accuracy_rf = accuracy_score(self.y_test, self.y_pred_rf)
            precision_rf = precision_score(self.y_test, self.y_pred_rf, pos_label='Good')
            recall_rf = recall_score(self.y_test, self.y_pred_rf, pos_label='Good')
            f1_rf = f1_score(self.y_test, self.y_pred_rf, pos_label='Good')
            roc_auc_rf = roc_auc_score(self.y_test, self.rf.predict_proba(self.X_test)[:, 1])

            # Create a DataFrame to display the results
            results = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
                'Logistic Regression': [accuracy_log_reg, precision_log_reg, recall_log_reg, f1_log_reg, roc_auc_log_reg],
                'Random Forest': [accuracy_rf, precision_rf, recall_rf, f1_rf, roc_auc_rf]
            })

            # Display the results
            print(results)
        except Exception as e:
            print(f"Error evaluating models: {e}")

    def save_model(self, model, filename):
        try:
            joblib.dump(model, filename)
            print(f"Model saved successfully as {filename}")
        except Exception as e:
            print(f"Error saving model: {e}")