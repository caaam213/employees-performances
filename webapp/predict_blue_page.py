import json
import sys
import pandas as pd
import requests
sys.path.append("/")
from scripts.blue_employees_script import train_and_generate_pipeline
import streamlit as st


file_path = "../data/productivity_employees_blue/train_dataset.csv"
api_url = "http://serving-api:8080/predict/"

def is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except:
        return False

def predict_blue_page_display():
    st.title("Prediction - Blue company")

    df = pd.read_csv(file_path, sep=',')  

    team = st.number_input("Team", min_value=0, max_value=12, step=1)
    targeted_productivity = st.number_input("Targeted productivity", min_value=0.0, max_value=1.0, step=0.01)
    smv = st.number_input("Smv", min_value=0.0, max_value=100.0, step=0.1)
    wip= st.number_input("Wip", min_value=0.0, max_value=10000.0, step=0.1)
    over_time = st.number_input("Overtime", min_value=0.0, max_value=10000.0, step=0.1)
    incentive = st.number_input("Incentive", min_value=0.0, max_value=10000.0, step=0.1)
    idle_time = st.selectbox('Idle_time', df['idle_time'].unique())
    idle_men = st.number_input("Idle_men", min_value=0.0, max_value=10000.0, step=0.1)
    no_of_style_change = st.number_input("no_of_style_change", min_value=0, max_value=2, step=1)
    no_of_workers =  st.number_input("no_of_workers", min_value=0.0, max_value=10000.0, step=1.0)
    month =  st.number_input("month", min_value=0.0, max_value=10000.0, step=0.1)


    quarter_options = ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"]
    selected_quarter = st.selectbox("Select Quarter", quarter_options)

    department_options = ["finishing", "sweing"]
    selected_department = st.selectbox("Select Department", department_options)

    day_options = ["Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]
    selected_day = st.selectbox("Select Day", day_options)

    if st.button("Predict employee performances"):
        input_data = {
        "team": float(team) if is_json_serializable(team) else None,
        "targeted_productivity": float(targeted_productivity) if is_json_serializable(targeted_productivity) else None,
        "smv": float(smv) if is_json_serializable(smv) else None,
        "wip": float(wip) if is_json_serializable(wip) else None,
        "over_time": float(over_time) if is_json_serializable(over_time) else None,
        "incentive": float(incentive) if is_json_serializable(incentive) else None,
        "idle_time": float(idle_time) if is_json_serializable(idle_time) else None,
        "idle_men": float(idle_men) if is_json_serializable(idle_men) else None,
        "no_of_style_change": int(no_of_style_change) if is_json_serializable(no_of_style_change) else None,
        "no_of_workers": float(no_of_workers) if is_json_serializable(no_of_workers) else None,
        "month": float(month) if is_json_serializable(month) else None,
        "quarter_Quarter1": 1 if selected_quarter == "Quarter1" else 0,
        "quarter_Quarter2": 1 if selected_quarter == "Quarter2" else 0,
        "quarter_Quarter3": 1 if selected_quarter == "Quarter3" else 0,
        "quarter_Quarter4": 1 if selected_quarter == "Quarter4" else 0,
        "quarter_Quarter5": 1 if selected_quarter == "Quarter5" else 0,
        "department_finishing": 1 if selected_department == "finishing" else 0,
        "department_sweing": 1 if selected_department == "sweing" else 0,
        "day_Monday": 1 if selected_day == "Monday" else 0,
        "day_Saturday": 1 if selected_day == "Saturday" else 0,
        "day_Sunday": 1 if selected_day == "Sunday" else 0,
        "day_Thursday": 1 if selected_day == "Thursday" else 0,
        "day_Tuesday": 1 if selected_day == "Tuesday" else 0,
        "day_Wednesday": 1 if selected_day == "Wednesday" else 0
    }
        
        # Appel de l'API de serving
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
        response = requests.post(api_url+"blue", json=input_data)

        # Vérifier si la requête a réussi
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error for the query ({response.status_code}): {response.text}"

    except Exception as e:
        return f"An error occurred : {str(e)}"

def train_model():
    try:
        custom_score, mae = train_and_generate_pipeline()
        return f"Model is trained with an score of {custom_score} and mae of {mae}"
    except Exception as e:
        return f"An error occurred : {str(e)}"