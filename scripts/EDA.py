## Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import ipaddress

# Load Datasets
def load_datasets():
    credit_card_df = pd.read_csv('creditcard.csv')
    fraud_data_df = pd.read_csv('Fraud_Data.csv')
    ip_address_df = pd.read_csv('IpAddress_to_Country.csv')
    return credit_card_df, fraud_data_df, ip_address_df

# Function to load and analyze data
def analyze_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Data from {file_path}")
        print("\nHead of the DataFrame:")
        display(df.head())
        print("\nInfo of the DataFrame:")
        display(df.info())
        print("\nDescribe the DataFrame:")
        display(df.describe(include='all'))
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def check_missing_values(credit_card_df, fraud_data_df, ip_address_df):
    """
    Checks for missing values in the three DataFrames and displays the results.

    Args:
        credit_card_df (pd.DataFrame): The creditcard.csv DataFrame.
        fraud_data_df (pd.DataFrame): The Fraud_Data.csv DataFrame.
        ip_address_df (pd.DataFrame): The IpAddress_to_Country.csv DataFrame.
    """
    print("Missing Values in Credit Card DataFrame:")
    print(credit_card_df.isnull().sum())
    print("\nMissing Values in Fraud Data DataFrame:")
    print(fraud_data_df.isnull().sum())
    print("\nMissing Values in IP Address DataFrame:")
    print(ip_address_df.isnull().sum())


# Handle Missing Values
def handle_missing_values(df):
    return df.dropna()

# Data Cleaning
def clean_data(df):
    df = df.drop_duplicates()
    return df

# Correct Data Types
def correct_data_types(fraud_data_df):
    fraud_data_df['signup_time'] = pd.to_datetime(fraud_data_df['signup_time'])
    fraud_data_df['purchase_time'] = pd.to_datetime(fraud_data_df['purchase_time'])
    fraud_data_df['ip_address'] = fraud_data_df['ip_address'].astype(int)
    return fraud_data_df

def pdf_univariate(df):
    # Visualize distributions of numerical features
    for col in df.select_dtypes(include=np.number):
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

    # Visualize counts of categorical features
    for col in df.select_dtypes(include=object):
        plt.figure(figsize=(8, 6))
        sns.countplot(x=col, data=df)
        plt.title(f'Count of {col}')
        plt.show()

# Exploratory Data Analysis (EDA)
def univariate_analysis(df):
    return df.describe()

def bivariate_analysis(df, target_col):
    sns.pairplot(df, hue=target_col)
    plt.show()

def bivariate_analysis_1():
 plt.figure(figsize=(12, 6))
 sns.boxplot(x='class', y='purchase_value', data=fraud_data_df)
 plt.title('Purchase Value by Class')
 plt.show()
 
 def bivariate_analysis_2():
  plt.figure(figsize=(12, 6))
  sns.scatterplot(x='purchase_value', y='age', hue='class', data=fraud_data_df)
  plt.title('Purchase Value vs. Age')
  plt.show()
# Merge Datasets for Geolocation Analysis
def convert_ip_to_integer(ip_address_df):
    ip_address_df['lower_bound_ip_address'] = ip_address_df['lower_bound_ip_address'].apply(lambda x: int(ipaddress.ip_address(x)))
    ip_address_df['upper_bound_ip_address'] = ip_address_df['upper_bound_ip_address'].apply(lambda x: int(ipaddress.ip_address(x)))
    return ip_address_df

# Map IP addresses to countries
def ip_to_country(ip):
    ip = int(ip)
    country = ip_address_df[(ip_address_df['lower_bound_ip_address'] <= ip) &
                              (ip_address_df['upper_bound_ip_address'] >= ip)]
    if not country.empty:
        return country['country'].values[0]
    return 'unknown'
fraud_data_df['country'] = fraud_data_df['ip_address'].apply(ip_to_country)



# Feature Engineering
def transaction_features(fraud_data_df):
    fraud_data_df['transaction_frequency'] = fraud_data_df.groupby('user_id')['timestamp'].transform('count')
    fraud_data_df['transaction_velocity'] = fraud_data_df.groupby('user_id')['timestamp'].diff().fillna(0)
    return fraud_data_df

def time_based_features(fraud_data_df):
    fraud_data_df['timestamp'] = pd.to_datetime(fraud_data_df['timestamp'])
    fraud_data_df['hour_of_day'] = fraud_data_df['timestamp'].dt.hour
    fraud_data_df['day_of_week'] = fraud_data_df['timestamp'].dt.dayofweek
    return fraud_data_df

# Normalization and Scaling
def normalize_and_scale(df):
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df

# Encode Categorical Features
def encode_categorical_features(df):
    le = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = le.fit_transform(df[column])
    return df

# Main function to run all preprocessing steps
def main_preprocessing():
    # Load datasets
    credit_card_df, fraud_data_df, ip_address_df = load_datasets()
    # Analyze creditcard dataset
    print("Analyzing creditcard dataset")
    analyze_data('../data/creditcard.csv/creditcard.csv')
    # fraud_data_df 
    print("Analyzing fraud_data dataset")
    analyze_data('../data/Fraud_Data.csv')
    print("="*80)
    # ip_address_df 
    print("Analyzing IpAddress_to_Country dataset")
    analyze_data('../data/IpAddress_to_Country.csv')
    print("="*80)
    #Checks for missing values in the three DataFrames and displays the results.
    check_missing_values(credit_card_df, fraud_data_df, ip_address_df)
    # Handle missing values
    credit_card_df = handle_missing_values(credit_card_df)
    fraud_data_df = handle_missing_values(fraud_data_df)
    
    # Data cleaning
    credit_card_df = clean_data(credit_card_df)
    fraud_data_df = clean_data(fraud_data_df)
    
    # Correct data types
    fraud_data_df = correct_data_types(fraud_data_df)
    
    # Exploratory Data Analysis (EDA)
    print("Credit Card Data - Univariate Analysis:")
    print(univariate_analysis(credit_card_df))
    print("Fraud Data - Univariate Analysis:")
    print(univariate_analysis(fraud_data_df))
    pdf_univariate(fraud_data_df)
    pdf_univariate(credit_card_df)
    print("Credit Card Data - Bivariate Analysis:")
    bivariate_analysis(credit_card_df, 'Class')
    print("Fraud Data - Bivariate Analysis:")
    bivariate_analysis(fraud_data_df, 'class')
    
    # Merge datasets for geolocation analysis
    ip_address_df = convert_ip_to_integer(ip_address_df)
    fraud_data_df = ip_to_country()
    
    # Feature engineering
    fraud_data_df = transaction_features(fraud_data_df)
    fraud_data_df = time_based_features(fraud_data_df)
    
    # Normalization and scaling
    credit_card_df = normalize_and_scale(credit_card_df)
    fraud_data_df = normalize_and_scale(fraud_data_df.select_dtypes(include=[np.number]))
    
    # Encode categorical features
    fraud_data_df = encode_categorical_features(fraud_data_df)
    
    return credit_card_df, fraud_data_df

# Execute preprocessing
credit_card_df, fraud_data_df = main_preprocessing()
# Save preprocessed datasets
fraud_data_df.to_csv('preprocessed_fraud_data.csv', index=False)
credit_card_df.to_csv('preprocessed_creditcard_data.csv', index=False)