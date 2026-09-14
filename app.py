import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="EduPredict AI | Student Risk Engine",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
<style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: bold;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -10px rgba(99, 102, 241, 0.5);
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
  try:
    return joblib.load("student_dropout_model.pkl")
  except Exception:
    return None


model = load_model()

st.title("🎓 EduPredict AI: Institutional Retention & Dropout Risk Analyzer")
st.markdown(
    "Client-ready interactive decision support dashboard powered by a direct"
    " Scikit-Learn pipeline."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Background & Profile",
    "📚 1st Semester Records",
    "📊 2nd Semester Records",
    "📈 Macro-Economics",
])

with tab1:
  st.subheader("Demographic & Academic Background")
  col1, col2, col3 = st.columns(3)
  with col1:
    marital_status = st.selectbox(
        "Marital Status",
        options=[1, 2, 3, 4, 5, 6],
        format_func=lambda x: {
            1: "Single",
            2: "Married",
            3: "Widower",
            4: "Divorced",
            5: "Facto union",
            6: "Legally separated",
        }[x],
    )
    application_mode = st.number_input("Application Mode", value=17)
    application_order = st.slider("Application Order", 1, 9, 1)
    course = st.number_input("Course Code", value=33)
    daytime_evening_attendance = st.selectbox(
        "Attendance Type",
        options=[1, 0],
        format_func=lambda x: "Daytime" if x == 1 else "Evening",
    )
  with col2:
    previous_qualification = st.number_input(
        "Previous Qualification Code", value=1
    )
    previous_qualification_grade = st.number_input(
        "Previous Qualification Grade", value=120.0
    )
    nacionality = st.number_input("Nationality Code", value=1)
    mothers_qualification = st.number_input("Mother's Qualification", value=1)
    fathers_qualification = st.number_input("Father's Qualification", value=1)
  with col3:
    mothers_occupation = st.number_input("Mother's Occupation", value=3)
    fathers_occupation = st.number_input("Father's Occupation", value=3)
    admission_grade = st.number_input("Admission Grade", value=120.0)
    displacement = st.selectbox(
        "Displaced Student",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    educational_special_needs = st.selectbox(
        "Educational Special Needs",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )

with tab2:
  st.subheader("First Semester Performance")
  col1, col2, col3 = st.columns(3)
  with col1:
    debtor = st.selectbox(
        "Debtor Status",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
        key="deb",
    )
    tuition_fees_up_to_date = st.selectbox(
        "Tuition Fees Up To Date",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    gender = st.selectbox(
        "Gender",
        options=[1, 0],
        format_func=lambda x: "Male" if x == 1 else "Female",
    )
  with col2:
    scholarship_holder = st.selectbox(
        "Scholarship Holder",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    age_at_enrollment = st.number_input("Age at Enrollment", value=20)
    international = st.selectbox(
        "International Student",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    cu_1st_sem_credited = st.number_input("1st Sem Credited Units", value=0)
  with col3:
    cu_1st_sem_enrolled = st.number_input("1st Sem Enrolled Units", value=6)
    cu_1st_sem_evaluations = st.number_input("1st Sem Evaluations", value=6)
    cu_1st_sem_approved = st.number_input("1st Sem Approved Units", value=5)
    cu_1st_sem_grade = st.number_input("1st Sem Grade Average", value=12.0)
    cu_1st_sem_without_evaluations = st.number_input(
        "1st Sem Units w/o Evaluations", value=0
    )

with tab3:
  st.subheader("Second Semester Performance")
  col1, col2 = st.columns(2)
  with col1:
    cu_2nd_sem_credited = st.number_input("2nd Sem Credited Units", value=0)
    cu_2nd_sem_enrolled = st.number_input("2nd Sem Enrolled Units", value=6)
    cu_2nd_sem_evaluations = st.number_input("2nd Sem Evaluations", value=6)
  with col2:
    cu_2nd_sem_approved = st.number_input("2nd Sem Approved Units", value=5)
    cu_2nd_sem_grade = st.number_input("2nd Sem Grade Average", value=12.0)
    cu_2nd_sem_without_evaluations = st.number_input(
        "2nd Sem Units w/o Evaluations", value=0
    )

with tab4:
  st.subheader("Macro-Economic Indicators")
  col1, col2, col3 = st.columns(3)
  with col1:
    unemployment_rate = st.number_input("Unemployment Rate (%)", value=10.8)
  with col2:
    inflation_rate = st.number_input("Inflation Rate (%)", value=1.4)
  with col3:
    gdp = st.number_input("GDP Growth Rate", value=1.74)

st.markdown("---")

if st.button("🚀 Run Predictive Analytics Engine"):
  if model is None:
    st.error(
        "⚠️ Model file (`student_dropout_model.pkl`) not found in root"
        " directory."
    )
  else:
    payload = {
        "Marital status": marital_status,
        "Application mode": application_mode,
        "Application order": application_order,
        "Course": course,
        "Daytime/evening attendance": daytime_evening_attendance,
        "Previous qualification": previous_qualification,
        "Previous qualification (grade)": previous_qualification_grade,
        "Nacionality": nacionality,
        "Mother's qualification": mothers_qualification,
        "Father's qualification": fathers_qualification,
        "Mother's occupation": mothers_occupation,
        "Father's occupation": fathers_occupation,
        "Admission grade": admission_grade,
        "Displaced": displacement,
        "Educational special needs": educational_special_needs,
        "Debtor": debtor,
        "Tuition fees up to date": tuition_fees_up_to_date,
        "Gender": gender,
        "Scholarship holder": scholarship_holder,
        "Age at enrollment": age_at_enrollment,
        "International": international,
        "Curricular units 1st sem (credited)": cu_1st_sem_credited,
        "Curricular units 1st sem (enrolled)": cu_1st_sem_enrolled,
        "Curricular units 1st sem (evaluations)": cu_1st_sem_evaluations,
        "Curricular units 1st sem (approved)": cu_1st_sem_approved,
        "Curricular units 1st sem (grade)": cu_1st_sem_grade,
        "Curricular units 1st sem (without evaluations)": (
            cu_1st_sem_without_evaluations
        ),
        "Curricular units 2nd sem (credited)": cu_2nd_sem_credited,
        "Curricular units 2nd sem (enrolled)": cu_2nd_sem_enrolled,
        "Curricular units 2nd sem (evaluations)": cu_2nd_sem_evaluations,
        "Curricular units 2nd sem (approved)": cu_2nd_sem_approved,
        "Curricular units 2nd sem (grade)": cu_2nd_sem_grade,
        "Curricular units 2nd sem (without evaluations)": (
            cu_2nd_sem_without_evaluations
        ),
        "Unemployment rate": unemployment_rate,
        "Inflation rate": inflation_rate,
        "GDP": gdp,
    }

    input_df = pd.DataFrame([payload])

    with st.spinner("Processing features through direct ML inference..."):
      try:
        pred = model.predict(input_df)[0]
        proba_arr = (
            model.predict_proba(input_df)
            if hasattr(model, "predict_proba")
            else None
        )
        prob = max(proba_arr[0]) if proba_arr is not None else 0.0

        if pred == 1:
          status = "High Risk of Dropout"
        else:
          status = "Enrolled / Low Risk"

        st.markdown("### 📋 Prediction Diagnostic Report")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
          if pred == 1:
            st.error(f"⚠️ **{status}**")
          else:
            st.success(f"✅ **{status}**")
            st.balloons()
        with res_col2:
          st.metric(
              label="Model Confidence Probability", value=f"{prob * 100:.2f}%"
          )
      except Exception as e:
        st.error(f"Error during prediction: {e}")
