import sys
import pandas as pd
import streamlit as st
import requests
sys.path.append("/")
from scripts.green_employees_script import train_and_generate_pipeline

file_path = "../data/productivity_employees_green/employee_data_performances.csv"
api_url = "http://serving-api:8080/predict/"

def predict_green_page_display():
    st.title("Prediction - Green company")

    df = pd.read_csv(file_path, sep=',')  
    single_individual = st.checkbox("Predict for a single person")

    input_data = {}

    fields = ['BusinessUnit', 'EmployeeStatus', 'EmployeeType', 'PayZone', 
          'EmployeeClassificationType', 'TerminationType', 'DepartmentType', 
          'Division', 'State', 'JobFunctionDescription', 'GenderCode','Title', 
          'RaceDesc', 'MaritalDesc', 'Performance Score']

    for field in fields:
        options = df[field].unique().tolist()
        if single_individual:
            selected_value = st.selectbox(field, options, key=f"{field}_selectbox")
            input_data[field] = selected_value
        else:
            options.insert(0, "All")
            selected_value = st.selectbox(field, options, key=f"{field}_selectbox_all")
            input_data[field] = selected_value

    
    if single_individual:
        start_date = st.number_input('Start Year', min_value=2019, max_value=2024, value=2019)
        dob = st.number_input('Date of birth', min_value=1900, max_value=2024, value=2000)
        input_data['StartDate'] = start_date
        input_data['DOB'] = dob
    else:
        min_start_date = st.number_input('Minimum Start Year', min_value=2018, max_value=2024, value=2018)
        max_start_date = st.number_input('Maximum Start Year', min_value=2018, max_value=2023, value=2023)

        min_dob = st.number_input('Minimum Date of Birth (Year)', min_value=1950, max_value=2024, value=1950)
        max_dob = st.number_input('Maximum Date of Birth (Year)', min_value=1950, max_value=2024, value=2024)

        input_data['StartDateMin'] = min_start_date
        input_data['StartDateMax'] = max_start_date
        input_data['DOBMin'] = min_dob
        input_data['DOBMax'] = max_dob

    if st.button("Predict employee performances"):
        
        input_data["SingleIndividual"] = 1 if single_individual else 0
        
        if not single_individual:
            input_data["PerformanceScore"] = input_data["Performance Score"]
        else:
            input_data["PerformanceScore"] = input_data["Performance Score"]
        
            
        prediction = get_prediction(input_data)

        # Affichage de la prédiction
        st.subheader("Prediction result:")
        st.write(prediction)

        
    
    # Bouton pour entraîner le modèle
    if st.button("Train model"):
        training_result = train_model()
        st.write(training_result)

def get_prediction(input_data):
    try:
        response = requests.post(api_url+"green", json=input_data)

        if response.status_code == 200:
            return response.json()
        else:
            return f"Error for the query ({response.status_code}): {response.text}"

    except Exception as e:
        return f"An error occurred : {str(e)}"

def train_model():
    try:
        accuracy = train_and_generate_pipeline()
        return f"Model is trained with an accuracy of {accuracy}"
    except Exception as e:
        return f"An error occurred : {str(e)}"