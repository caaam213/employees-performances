import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin


file_path = "../data/productivity_employees_green/employee_data_performances.csv"
file_path_predict = "../artifacts/predict_green_performance_pipeline"

def extract_year(date_string):
    if date_string != "?" and date_string is not np.nan:
        return int(str(date_string).split("-")[-1])
    return -1

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freqs = {}

    def fit(self, X_, y=None):
        for column in X_.columns:
            self.freqs[column] = X_[column].value_counts(normalize=True)
        return self

    def transform(self, X):
        X_copy = X.copy()
        for column in X.columns:
            X_copy[column] = X[column].map(self.freqs[column])
        return X_copy


def load_data(file_path_):
    dataframe = pd.read_csv(file_path_, sep=',')  
    dataframe['DOB'] = dataframe['DOB'].apply(extract_year)
    dataframe['StartDate'] = dataframe['StartDate'].apply(extract_year)
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1].astype(int)
    return dataframe, X, y

def custom_scoring(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return (accuracy + precision) / 2


def create_pipeline(steps, X_, y_, pipeline_file):
    pipeline = Pipeline(steps)
    pipeline.fit(X_, y_)
    with open(pipeline_file+'.pkl', 'wb') as file:
        pickle.dump(pipeline, file)

def load_and_predict(X_test_, pipeline_file):
    with open(pipeline_file+'.pkl', 'rb') as file:
        loaded_pipeline = pickle.load(file)
    predictions = loaded_pipeline.predict(X_test_)
    return predictions

def train_and_generate_pipeline(X):

    dataframe, X, y = load_data(file_path)
    y = np.where(y <= 3, 0, 1)

    best_model = RandomForestClassifier(min_samples_leaf=4, min_samples_split= 10, n_estimators= 500)
    

    cols = ['EmployeeStatus', 'EmployeeType', 'PayZone', 'EmployeeClassificationType', 'TerminationType', 'GenderCode', 'RaceDesc', 'MaritalDesc', 'Performance Score']
    cols_to_iterate = ['Title', 'BusinessUnit', 'DepartmentType', 'Division', 'State', 'JobFunctionDescription']
    encoder = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', encoder, cols),
            ('freq_encode', FrequencyEncoder(), cols_to_iterate)
        ],
        remainder='drop'

    )
    steps=[
        ('preprocessor', preprocessor),
        ('imputer', SimpleImputer(strategy='mean')), 
        ('normalisation', MinMaxScaler()),  
        ('classifier', best_model)  
    ]

    create_pipeline(steps, X, y, file_path_predict)
    predictions = load_and_predict(X, file_path_predict)

    accuracy = accuracy_score(y, predictions)

    return accuracy



def predict_green(input_data):
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data, index=[0])
    predictions = load_and_predict(input_data, file_path_predict)
    return predictions

def predict_with_less_features(selected_columns):
    df_emp_perf = pd.read_csv(file_path, sep=',')
    df_emp_perf['DOB'] = df_emp_perf['DOB'].apply(extract_year)
    df_emp_perf['StartDate'] = df_emp_perf['StartDate'].apply(extract_year)
    X = df_emp_perf.iloc[:, :-1]
    X = X[selected_columns]

    cols = ['EmployeeStatus', 'EmployeeType', 'PayZone', 'EmployeeClassificationType', 'TerminationType', 'GenderCode', 'RaceDesc', 'MaritalDesc', 'Performance Score']

    encoder = OneHotEncoder()

    

    non_encoded_cols = []

    for column in cols:
        if column not in X.columns:
            continue
        if X[column].dtype == 'object':
            encoder.fit(df_emp_perf[[column]])
            encoded_data = encoder.transform(df_emp_perf[[column]]).toarray()
            X_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
            X = pd.concat([X.drop(column, axis=1), X_encoded], axis=1)
        else:
            non_encoded_cols.append(column)

    X = pd.concat([X, df_emp_perf[non_encoded_cols]], axis=1)

    cols_to_iterate = ['Title', 'BusinessUnit', 'DepartmentType', 'Division', 'State', 'JobFunctionDescription']

    for column in cols_to_iterate:
        if column not in X.columns:
            continue
        freqs = X[column].value_counts(normalize=True)
        X[column] = X[column].map(freqs)

    

    y = df_emp_perf.iloc[:, -1].astype(int)
    y = np.where(y <= 3, 0, 1)

    best_model = RandomForestClassifier(min_samples_leaf=4, min_samples_split= 10, n_estimators= 500)

    best_model.fit(X, y)
    return best_model.predict(X)






