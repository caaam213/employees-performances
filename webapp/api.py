import pandas as pd
import streamlit as st
from fastapi import FastAPI
import sys
sys.path.append("/")
from webapp import predict_blue_page, predict_green_page, reporting_green


app = FastAPI()


# URL de l'API de serving
api_url = "http://serving-api:8080/predict/"

def main():
    page = st.sidebar.selectbox("Select a page", ["Accueil", "Prédiction - Green company", "Prédiction - Blue company", "Reporting Power BI - Green company"])

    if page == "Accueil":
        home()
    elif page == "Prédiction - Green company":
        predict_green_page.predict_green_page_display()
    elif page == "Prédiction - Blue company":
        predict_blue_page.predict_blue_page_display()
    elif page == "Reporting Power BI - Green company":
        reporting_green.reporting_green_display()


def home():
    st.title("Home")
    st.write("Welcome !")

    st.title("Welcome to Lumelia - Predicting Employee Well-being and Performance")
    st.write("Lumelia is a company specializing in workplace well-being and enhancing employee performance.")
    st.write("The purpose of this site is to make predictions based on employee profiles for two different companies.")
    st.write("Authors : MERAOUI Camélia - PERVENCHE Clémence - ROCHER Ludovic")

if __name__ == "__main__":
    main()
