
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd

model = joblib.load('student_dropout_model.pkl')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class StudentData(BaseModel):
    marital_status: int
    application_mode: int
    application_order: int
    course: int
    daytime_evening_attendance: int
    previous_qualification: int
    previous_qualification_grade: float
    nacionality: int
    mothers_qualification: int
    fathers_qualification: int
    mothers_occupation: int
    fathers_occupation: int
    admission_grade: float
    displaced: int
    educational_special_needs: int
    debtor: int
    tuition_fees_up_to_date: int
    gender: int
    scholarship_holder: int
    age_at_enrollment: int
    international: int
    cu_1st_sem_credited: int
    cu_1st_sem_enrolled: int
    cu_1st_sem_evaluations: int
    cu_1st_sem_approved: int
    cu_1st_sem_grade: float
    cu_1st_sem_without_evaluations: int
    cu_2nd_sem_credited: int
    cu_2nd_sem_enrolled: int
    cu_2nd_sem_evaluations: int
    cu_2nd_sem_approved: int
    cu_2nd_sem_grade: float
    cu_2nd_sem_without_evaluations: int
    unemployment_rate: float
    inflation_rate: float
    gdp: float

class PredictionResponse(BaseModel):
    prediction: int
    dropout_probability: float
    risk_status: str

@app.get('/')
def greet():
    return {'message': 'Student Dropout Prediction API is active'}

@app.post('/predict', response_model=PredictionResponse)
def predict(data: StudentData):
    input_row = pd.DataFrame([{
        'Marital status': data.marital_status,
        'Application mode': data.application_mode,
        'Application order': data.application_order,
        'Course': data.course,
        'Daytime/evening attendance': data.daytime_evening_attendance,
        'Previous qualification': data.previous_qualification,
        'Previous qualification (grade)': data.previous_qualification_grade,
        'Nacionality': data.nacionality,
        "Mother's qualification": data.mothers_qualification,
        "Father's qualification": data.fathers_qualification,
        "Mother's occupation": data.mothers_occupation,
        "Father's occupation": data.fathers_occupation,
        'Admission grade': data.admission_grade,
        'Displaced': data.displaced,
        'Educational special needs': data.educational_special_needs,
        'Debtor': data.debtor,
        'Tuition fees up to date': data.tuition_fees_up_to_date,
        'Gender': data.gender,
        'Scholarship holder': data.scholarship_holder,
        'Age at enrollment': data.age_at_enrollment,
        'International': data.international,
        'Curricular units 1st sem (credited)': data.cu_1st_sem_credited,
        'Curricular units 1st sem (enrolled)': data.cu_1st_sem_enrolled,
        'Curricular units 1st sem (evaluations)': data.cu_1st_sem_evaluations,
        'Curricular units 1st sem (approved)': data.cu_1st_sem_approved,
        'Curricular units 1st sem (grade)': data.cu_1st_sem_grade,
        'Curricular units 1st sem (without evaluations)': data.cu_1st_sem_without_evaluations,
        'Curricular units 2nd sem (credited)': data.cu_2nd_sem_credited,
        'Curricular units 2nd sem (enrolled)': data.cu_2nd_sem_enrolled,
        'Curricular units 2nd sem (evaluations)': data.cu_2nd_sem_evaluations,
        'Curricular units 2nd sem (approved)': data.cu_2nd_sem_approved,
        'Curricular units 2nd sem (grade)': data.cu_2nd_sem_grade,
        'Curricular units 2nd sem (without evaluations)': data.cu_2nd_sem_without_evaluations,
        'Unemployment rate': data.unemployment_rate,
        'Inflation rate': data.inflation_rate,
        'GDP': data.gdp
    }])
    
    prediction = int(model.predict(input_row)[0])
    probability = float(model.predict_proba(input_row)[0][1])
    status = "High Risk of Dropout" if prediction == 1 else "Low Risk of Dropout"
    
    return PredictionResponse(
        prediction=prediction,
        dropout_probability=round(probability, 4),
        risk_status=status
    )