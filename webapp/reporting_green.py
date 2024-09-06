from datetime import datetime
import sys
import pandas as pd
import requests
sys.path.append("/")
from scripts.green_employees_script import load_data, predict_green, predict_with_less_features

from scripts.blue_employees_script import train_and_generate_pipeline
import streamlit as st

file_path = "../data/productivity_employees_green/employee_data_performances.csv"
file_path_reporting = "../reporting/"


def reporting_green_display():
    # TODO : Déplacer le code s'il y a le temps 
    st.title("Reporting Power BI - Green company")

    df, _, _ = load_data(file_path)
    X = df.iloc[:, :-1]
    selected_columns = ["EmpID"]

    for column in X.columns.drop(['EmpID', 'FirstName', 'LastName', 'ExitDate', 'Supervisor', 'ADEmail', 'TerminationDescription', 'LocationCode' ], errors='ignore'):
        selected = st.checkbox(column)
        if selected:
            selected_columns.append(column)

    # Bouton pour valider la sélection
    if st.button("Valider la sélection"):
        predictions =  predict_with_less_features(selected_columns)
        X = X[selected_columns]
        X['prediction'] = predictions
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{file_path_reporting}{current_datetime}_reporting_powerbi_green.csv"
        X.to_csv(filename, index=False)

        st.write(f"File created with name {filename}")

