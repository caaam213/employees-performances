from datetime import datetime
import sys
from typing import Union
import pandas as pd
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel, validator
sys.path.append("/")
from scripts.blue_employees_script import predict_blue
from scripts.green_employees_script import load_data, predict_green

file_path = "../data/productivity_employees_green/employee_data_performances.csv"
file_path_reporting = "../reporting/"

class EmployeeDataGreen(BaseModel):
    EmpID: str = ""
    FirstName: str = ""
    LastName: str = ""
    StartDate: int = 0
    StartDateMin:int = 0
    StartDateMax:int = 0
    ExitDate: str = ""
    Title: str
    Supervisor: str = ""
    ADEmail: str = ""
    BusinessUnit: str
    EmployeeStatus: str
    EmployeeType: str
    PayZone: str
    EmployeeClassificationType: str
    TerminationType: str
    TerminationDescription: str = ""
    DepartmentType: str
    Division: str
    DOB: int = 0
    DOBMin:int = 0
    DOBMax:int = 0
    State: str
    JobFunctionDescription: str
    GenderCode: str
    LocationCode: str = ""
    RaceDesc: str
    MaritalDesc: str
    PerformanceScore: str
    SingleIndividual : int

    @validator('StartDate')
    def parse_start_date(cls, start_date: int) -> int:
        return int(str(start_date)[-2:])

    @validator('StartDateMin')
    def parse_StartDateMin(cls, start_date: int) -> int:
        return int(str(start_date)[-2:])

    @validator('StartDateMax')
    def parse_StartDateMax(cls, start_date: int) -> int:
        return int(str(start_date)[-2:])


class EmployeeDataBlue(BaseModel):
    team: float
    targeted_productivity: float
    smv: float
    wip: float
    over_time: float
    incentive: float
    idle_time: float
    idle_men: float
    no_of_style_change: int
    no_of_workers: float
    month: float
    quarter_Quarter1: int
    quarter_Quarter2: int
    quarter_Quarter3: int
    quarter_Quarter4: int
    quarter_Quarter5: int
    department_finishing: int
    department_sweing: int
    day_Monday: int
    day_Saturday: int
    day_Sunday: int
    day_Thursday: int
    day_Tuesday: int
    day_Wednesday: int


app = FastAPI()

def insert_at_position(d, key, value, position):
    new_dict = {}
    for i, (k, v) in enumerate(d.items()):
        if i == position:
            new_dict[key] = value
        new_dict[k] = v
    if position == len(d):
        new_dict[key] = value
    return new_dict

@app.post("/predict/green")
def predict_performances_green(input_data: EmployeeDataGreen):
    
    if input_data.SingleIndividual == 1:
        input_data = input_data.dict()
        input_data["Performance Score"] = input_data["PerformanceScore"]
        del input_data["PerformanceScore"]
        del input_data["StartDateMin"]
        del input_data["StartDateMax"]
        del input_data["DOBMin"]
        del input_data["DOBMax"]
        
    else:
        columns_to_keep = ["EmpID"]
        df, X, y = load_data(file_path)
        X = X[(X['StartDate'] >= input_data.StartDateMin) & (X['StartDate'] <= input_data.StartDateMax)]
        X = X[(X['DOB'] >= input_data.DOBMin) & (X['DOB'] <= input_data.DOBMax)]
        if input_data.PerformanceScore != "All" and input_data.PerformanceScore != "":
            X = X[X['Performance Score'] == input_data.PerformanceScore]
            columns_to_keep.append('Performance Score')
        
        for field in X.columns.drop(['StartDate', 'DOB', 'Performance Score']):
            if getattr(input_data, field) != "All" and getattr(input_data, field) != "":
                X = X[X[field] == getattr(input_data, field)]
                columns_to_keep.append(field)
            print(field)
            print(X)
        input_data = X
            
    print("Input Data:", input_data)
    print("Input Data Type:", type(input_data))

    try:
        prediction_result = predict_green(input_data=input_data)

        if len(prediction_result) == 1:
            result_message = "This employee will be efficient" if int(prediction_result[0]) == 1 else "This employee will not be efficient"
            return {
                "prediction_result": str(prediction_result[0]),
                "message": result_message
            }
        else:
            
            X = X[columns_to_keep]
            X['prediction'] = prediction_result
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{file_path_reporting}{current_datetime}_reporting_green.csv"
            X.to_csv(filename, index=False)

            return {
                "Moyenne des prédictions": str(np.mean(prediction_result))
            }
    except Exception as e:
        print("Exception:", e)
        return {"error": f"An error occurred during prediction : {str(e)}"}

@app.post("/predict/blue")
def predict_performances_blue(employee_data: EmployeeDataBlue):
    input_data = employee_data.dict()
    input_data = insert_at_position(input_data, "department_finishing ", input_data["department_finishing"], 17)

    print("Input Data:", input_data)
    print("Input Data Type:", type(input_data))

    try:
        prediction_result = predict_blue(input_data=input_data)[0]
        return {
            "employe_performance": str(prediction_result),
        }
    except Exception as e:
        print("Exception:", e)
        return {"error": "An error occurred during prediction."}


