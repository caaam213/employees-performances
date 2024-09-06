# Employee Performance Prediction API
This repository contains a school project developed by a team of three, aiming to predict employee performance using supervised learning algorithms. The project leverages two datasets and provides a simple API for predictions using Docker and Streamlit.

# Introduction
The goal of this project is to predict employee performance using supervised algorithms for linear regression tasks. We implemented an API that enables users to interact with our predictive model through a web interface built with Streamlit, all containerized using Docker.

# Project Structure
The main components of this project include:

- API: A RESTful API to interact with the predictive model.
- Streamlit App: A simple user interface for testing predictions.
- Docker: Containerization for easy deployment and reproducibility.

# Datasets
We used two different datasets for training our models:

- Dataset 1: [Employee Dataset(All in One)](https://www.kaggle.com/datasets/ravindrasinghrana/employeedataset)
- Dataset 2: [Employee Performance Prediction](https://www.kaggle.com/datasets/gauravduttakiit/employee-performance-prediction)


# Algorithms Used
The project applies supervised learning algorithms for linear problems. Some of the models used include:

- AdaBoostRegressor2
- LassoRegressor
- ElasticNetRegressor

To calculate the model performances, we use those metrics: 
- MAE
- RMSE
- R2 score
- Custom score : This custom scoring function calculates the proportion of predicted values that fall within an error margin of 0.1 from the true values. It returns a score representing the percentage of predictions that meet this accuracy threshold.

# Contributors
This project was developed by:
- MERAOUI Camélia
- PERVENCHE Clémence
- ROCHER Ludovic
