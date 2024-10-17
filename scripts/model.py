import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import mlflow
import mlflow.sklearn

# Load your datasets
# Load preprocessed data
pre_credit_data = pd.read_csv('../data/preprocessed_creditcard_data.csv')
pre_fraud_data_df = pd.read_csv('../data/preprocessed_fraud_data.csv')
# Feature and target separation
X_credit = pre_credit_data.drop(columns=['Class'])
y_credit = pre_credit_data['Class']

X_fraud = pre_fraud_data_df.drop(columns=['class'])
y_fraud = pre_fraud_data_df['class']

# Function to extract datetime features
def extract_datetime_features(df, datetime_column):
    df_copy = df.copy()
    df_copy[datetime_column] = pd.to_datetime(df_copy[datetime_column])
    df_copy['year'] = df_copy[datetime_column].dt.year
    df_copy['month'] = df_copy[datetime_column].dt.month
    df_copy['day'] = df_copy[datetime_column].dt.day
    df_copy['hour'] = df_copy[datetime_column].dt.hour
    df_copy = df_copy.drop(columns=[datetime_column])
    return df_copy

# Preprocess fraud data (assuming 'datetime_column' is the name of your datetime column)
datetime_column = 'transaction_date'  # Replace with the actual datetime column name if it exists
if datetime_column in X_fraud.columns:
    X_fraud = extract_datetime_features(X_fraud, datetime_column)

# Identify numeric and categorical columns
numeric_features = X_fraud.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_fraud.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipeline for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the data
X_fraud = preprocessor.fit_transform(X_fraud)

# Train-test split
X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)
X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP": MLPClassifier(max_iter=1000)
}

# Helper function to train and evaluate models
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    print(classification_report(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_prob))
    
    return model

# Function to log experiments using MLflow
def log_experiment(model_name, model, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)
        
        mlflow.sklearn.log_model(model, "model")
        
        print(f"{model_name} logged successfully.")

# Evaluate and log models for credit card data
print("Evaluating models for credit card data:")
for name, model in models.items():
    print(f"\n{name}")
    trained_model = train_and_evaluate(model, X_credit_train, y_credit_train, X_credit_test, y_credit_test)
    log_experiment(name, trained_model, X_credit_train, y_credit_train, X_credit_test, y_credit_test)

# Evaluate and log models for fraud data
print("\nEvaluating models for fraud data:")
for name, model in models.items():
    print(f"\n{name}")
    trained_model = train_and_evaluate(model, X_fraud_train, y_fraud_train, X_fraud_test, y_fraud_test)
    log_experiment(name, trained_model, X_fraud_train, y_fraud_train, X_fraud_test, y_fraud_test)