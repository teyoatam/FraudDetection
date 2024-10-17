import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import ipaddress

from scrpits.EDA import clean_data, convert_ip_to_integer, correct_data_types, encode_categorical_features, handle_missing_values, ip_to_country, normalize_and_scale, time_based_features, transaction_features

# ... (Your existing code for functions: load_datasets, analyze_data, check_missing_values, 
#         handle_missing_values, clean_data, correct_data_types, pdf_univariate, 
#         univariate_analysis, bivariate_analysis, convert_ip_to_integer, ip_to_country, 
#         transaction_features, time_based_features, normalize_and_scale, encode_categorical_features,
#         main_preprocessing) ...

class TestPreprocessing(unittest.TestCase):

    def test_load_datasets(self):
        # Mock data for testing
        credit_card_df = pd.DataFrame({'col1': [1, 2, 3]})
        fraud_data_df = pd.DataFrame({'col2': [4, 5, 6]})
        ip_address_df = pd.DataFrame({'col3': [7, 8, 9]})

        # Replace the actual file reading with mock data
        def mock_read_csv(file_path):
            if file_path == 'creditcard.csv':
                return credit_card_df
            elif file_path == 'Fraud_Data.csv':
                return fraud_data_df
            elif file_path == 'IpAddress_to_Country.csv':
                return ip_address_df
            else:
                raise FileNotFoundError

        # Patch the read_csv function
        with unittest.mock.patch('pandas.read_csv', side_effect=mock_read_csv):
            result_credit_card_df, result_fraud_data_df, result_ip_address_df = load_datasets()

        self.assertEqual(result_credit_card_df.equals(credit_card_df), True)
        self.assertEqual(result_fraud_data_df.equals(fraud_data_df), True)
        self.assertEqual(result_ip_address_df.equals(ip_address_df), True)

    def test_handle_missing_values(self):
        # Test with a DataFrame containing missing values
        df = pd.DataFrame({'col1': [1, np.nan, 3], 'col2': [4, 5, 6]})
        expected_df = pd.DataFrame({'col1': [1, 3], 'col2': [4, 6]})  # Dropped NaN rows
        result_df = handle_missing_values(df)
        self.assertTrue(result_df.equals(expected_df))

    def test_clean_data(self):
        # Test with a DataFrame containing duplicate rows
        df = pd.DataFrame({'col1': [1, 1, 2, 3], 'col2': [4, 4, 5, 6]})
        expected_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        result_df = clean_data(df)
        self.assertTrue(result_df.equals(expected_df))

    def test_correct_data_types(self):
        # Test with a DataFrame with incorrect data types
        df = pd.DataFrame({'signup_time': ['2023-01-01', '2023-01-02'],
                           'purchase_time': ['2023-01-03', '2023-01-04'],
                           'ip_address': ['1.2.3.4', '5.6.7.8']})
        expected_df = pd.DataFrame({'signup_time': pd.to_datetime(['2023-01-01', '2023-01-02']),
                                   'purchase_time': pd.to_datetime(['2023-01-03', '2023-01-04']),
                                   'ip_address': [int(ipaddress.ip_address('1.2.3.4')),
                                                  int(ipaddress.ip_address('5.6.7.8'))]})
        result_df = correct_data_types(df)
        self.assertTrue(result_df.equals(expected_df))

    def test_convert_ip_to_integer(self):
        # Test with a DataFrame containing IP addresses as strings
        df = pd.DataFrame({'lower_bound_ip_address': ['1.2.3.4', '5.6.7.8'],
                           'upper_bound_ip_address': ['9.10.11.12', '13.14.15.16']})
        expected_df = pd.DataFrame({'lower_bound_ip_address': [int(ipaddress.ip_address('1.2.3.4')),
                                                              int(ipaddress.ip_address('5.6.7.8'))],
                                   'upper_bound_ip_address': [int(ipaddress.ip_address('9.10.11.12')),
                                                              int(ipaddress.ip_address('13.14.15.16'))]})
        result_df = convert_ip_to_integer(df)
        self.assertTrue(result_df.equals(expected_df))

    def test_ip_to_country(self):
        # Mock ip_address_df for testing
        ip_address_df = pd.DataFrame({'lower_bound_ip_address': [int(ipaddress.ip_address('1.2.3.4')),
                                                                int(ipaddress.ip_address('5.6.7.8'))],
                                   'upper_bound_ip_address': [int(ipaddress.ip_address('9.10.11.12')),
                                                                int(ipaddress.ip_address('13.14.15.16'))],
                                   'country': ['USA', 'Canada']})
        # Test with an IP address within the range
        self.assertEqual(ip_to_country(int(ipaddress.ip_address('2.2.2.2'))), 'USA')
        # Test with an IP address outside the range
        self.assertEqual(ip_to_country(int(ipaddress.ip_address('1.1.1.1'))), 'unknown')

    def test_transaction_features(self):
        # Test with a DataFrame containing transaction data
        df = pd.DataFrame({'user_id': [1, 1, 2, 2], 'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        expected_df = pd.DataFrame({'user_id': [1, 1, 2, 2], 'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
                                   'transaction_frequency': [2, 2, 2, 2], 'transaction_velocity': [pd.Timedelta(days=1), pd.Timedelta(days=1), pd.Timedelta(days=1), pd.Timedelta(days=1)]})
        result_df = transaction_features(df)
        self.assertTrue(result_df.equals(expected_df))

    def test_time_based_features(self):
        # Test with a DataFrame containing timestamps
        df = pd.DataFrame({'timestamp': ['2023-01-01 10:00:00', '2023-01-02 14:00:00']})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        expected_df = pd.DataFrame({'timestamp': ['2023-01-01 10:00:00', '2023-01-02 14:00:00'],
                                   'hour_of_day': [10, 14], 'day_of_week': [0, 1]})
        result_df = time_based_features(df)
        self.assertTrue(result_df.equals(expected_df))

        def test_normalize_and_scale(self):
          """Test normalization and scaling."""
        test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        expected_df = pd.DataFrame({'A': [-1.22474487, 0, 1.22474487], 'B': [-1.22474487, 0, 1.22474487]})
        result_df = normalize_and_scale(test_df)
        np.testing.assert_array_almost_equal(result_df.values, expected_df.values)

    