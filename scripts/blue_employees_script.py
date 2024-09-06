import pandas as pd
from sklearn.pipeline import Pipeline
import pickle
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error



file_path = "../data/productivity_employees_blue/train_dataset.csv"
file_path_predict = "../artifacts/predict_blue_performance_pipeline"

def load_data(file_path_):
    dataframe = pd.read_csv(file_path_, sep=',')  
    X = dataframe.drop(columns=["actual_productivity"], inplace=False)
    y = dataframe["actual_productivity"]
    y = y.astype(float)
    return dataframe, X, y


def custom_scoring_function(y_true, y_pred):
    error_margin = 0.1  
    
    absolute_errors = abs(y_true - y_pred)
    
    within_margin = (absolute_errors <= error_margin).mean()
    
    return within_margin

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

def train_and_generate_pipeline():

    dataframe, X, y = load_data(file_path)
    best_model = TransformedTargetRegressor(regressor=RandomForestRegressor(max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100))

    steps=[
        ('imputer', SimpleImputer(strategy='mean')), 
        ('normalisation', MinMaxScaler()),  
        ('classifier', best_model)  
    ]
    create_pipeline(steps, X, y, file_path_predict)
    predictions = load_and_predict(X, file_path_predict)
    custom_score = custom_scoring_function(X, predictions)
    mae = mean_absolute_error(X, predictions)

    print(f"Score : {custom_score}")
    print(f"Mae : {mae}")

    return custom_score, mae

def predict_blue(input_data):
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data, index=[0])
    predictions = load_and_predict(input_data, file_path_predict)
    return predictions
