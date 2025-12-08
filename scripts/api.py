from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from eda import data_eda
from enum import Enum

app = FastAPI()
model = joblib.load("../deploy_model/XGB.joblib")

class DepartmentEnum(str, Enum):
    sales = "sales"
    accouting="accounting"
    hr = "hr"
    IT ="IT"
    randd="randd"
    product_mng="product_mng"
    marketing="marketing"
    technical = "technical"
    support = "support"
    management = "management"

class SalaryEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class InputData(BaseModel):
    satisfaction_level: float           
    last_evaluation: float              
    number_project: int                
    workload_status: bool = Field(..., description="1 = average monthly hours > 175, 0 = average monthly hours <= 175 ")
    time_spend_company: int
    work_accident: bool = Field(..., description="1 = yes, 0 = no")                
    promotion_last_5years: bool = Field(..., description="1 = have had promotions in the past 5 years, 0 = no")      
    department: DepartmentEnum
    salary: SalaryEnum
   
class encoder(data_eda):
    def __init__(self, model, expected_columns):
        self.df = None   
        self.model = model
        self.expected_columns = expected_columns  

    def encode_row(self, row: dict):
        self.df = pd.DataFrame([row])
        super().encode()  
        self.df = self.df.reindex(columns=self.expected_columns, fill_value=0)
        return self.df

expected_columns = ['satisfaction_level', 'last_evaluation', 'number_project', 'time_spend_company',
                    'work_accident', 'promotion_last_5years', 'salary', 'workload_status',
                    'department_accounting', 'department_hr', 'department_it', 'department_management',
                    'department_marketing', 'department_product_mng', 'department_randd', 'department_sales',
                    'department_support', 'department_technical']

encode = encoder(model, expected_columns)

@app.post("/predict_churn_rate_HR_department")

def predict(data: InputData):
    try:
        df = encode.encode_row(data.dict())
        pred = model.predict(df)[0]
        return {"prediction": int(pred)}
    except Exception as e:
        return {"error": str(e)}
