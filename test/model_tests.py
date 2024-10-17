import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Mock datasets for testing
mock_credit_data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'Class': np.random.randint(0, 2, 100)
})

mock_fraud_data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'transaction_date': pd.date_range(start='1/1/2020', periods=100, freq='D'),
    'class': np.random.randint(0, 2, 100)
})

mock_preprocessed_fraud_data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'year': 2020,
    'month': np.random.randint(1, 13, 100),
    'day': np.random.randint(1, 29, 100),
    'hour': np.random.randint(0, 24, 100),
    'class': np.random.randint(0, 2, 100)
})

class TestModelPipeline(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_data_loading(self, mock_read_csv):
        mock_read_csv.side_effect = [mock_credit_data, mock_preprocessed_fraud_data]

        pre_credit_data = pd.read_csv('../data/preprocessed_creditcard_data.csv')
        pre_fraud_data_df = pd.read_csv('../data/preprocessed_fraud_data.csv')

        self.assertEqual(len(pre_credit_data), 100)
        self.assertEqual(len(pre_fraud_data_df), 100)

    def test_extract_datetime_features(self):
        from model import extract_datetime_features  # Replace 'your_module' with the actual module name

        result = extract_datetime_features(mock_fraud_data, 'transaction_date')
        self.assertIn('year', result.columns)
        self.assertIn('month', result.columns)
        self.assertIn('day', result.columns)
        self.assertIn('hour', result.columns)

    def test_preprocessing_pipeline(self):
        from model import preprocessor  # Replace 'your_module' with the actual module name

        X_fraud = mock_preprocessed_fraud_data.drop(columns=['class'])
        X_transformed = preprocessor.fit_transform(X_fraud)

        self.assertEqual(X_transformed.shape[0], 100)

    @patch('mlflow.sklearn.log_model')
    @patch('mlflow.log_metric')
    @patch('mlflow.log_param')
    @patch('mlflow.start_run')
    def test_log_experiment(self, mock_start_run, mock_log_param, mock_log_metric, mock_log_model):
        from model import log_experiment  # Replace 'your_module' with the actual module name

        model = LogisticRegression()
        X_train, X_test, y_train, y_test = train_test_split(mock_credit_data.drop(columns=['Class']), mock_credit_data['Class'], test_size=0.2, random_state=42)

        log_experiment("Logistic Regression", model, X_train, y_train, X_test, y_test)

        self.assertTrue(mock_start_run.called)
        self.assertTrue(mock_log_param.called)
        self.assertTrue(mock_log_metric.called)
        self.assertTrue(mock_log_model.called)

    def test_train_and_evaluate(self):
        from model import train_and_evaluate  # Replace 'your_module' with the actual module name

        model = LogisticRegression()
        X_train, X_test, y_train, y_test = train_test_split(mock_credit_data.drop(columns=['Class']), mock_credit_data['Class'], test_size=0.2, random_state=42)

        trained_model = train_and_evaluate(model, X_train, y_train, X_test, y_test)

        self.assertIsInstance(trained_model, LogisticRegression)

if __name__ == '__main__':
    unittest.main()