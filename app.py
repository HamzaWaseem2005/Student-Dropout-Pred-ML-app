import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Dropout Prediction System", page_icon="🎓", layout="centered"
)


@st.cache_resource
def load_model():
  try:
    return joblib.load("student_dropout_model.pkl")
  except Exception as e:
    return None


model = load_model()

st.title("🎓 Student Dropout Prediction System")
st.markdown(
    "Enter the student's academic and demographic details below to predict"
    " whether they are at risk of dropping out."
)

st.markdown("---")

if model is None:
  st.error(
      "⚠️ Model file (`student_dropout_model.pkl`) could not be loaded. Please"
      " ensure it is present in the repository root."
  )
else:
  with st.form("prediction_form"):
    st.subheader("📝 Student Information")

    col1, col2 = st.columns(2)

    with col1:
      age = st.number_input(
          "Age at Enrollment", min_value=15, max_value=70, value=20
      )
      admission_grade = st.number_input(
          "Admission Grade", min_value=0.0, max_value=200.0, value=120.0
      )
      tuition_up_to_date = st.selectbox(
          "Tuition Fees Up To Date?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No"
      )

    with col2:
      scholarship_holder = st.selectbox(
          "Scholarship Holder?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No"
      )
      curricular_units_2nd_sem = st.number_input(
          "Curricular Units 2nd Sem (Approved)", min_value=0, max_value=20, value=5
      )
      curricular_units_1st_sem = st.number_input(
          "Curricular Units 1st Sem (Approved)", min_value=0, max_value=20, value=5
      )

    submitted = st.form_submit_button(
        "🚀 Run Predictive Analytics", use_container_width=True
    )

  if submitted:
    input_data = pd.DataFrame({
        "Age at enrollment": [age],
        "Admission grade": [admission_grade],
        "Tuition fees up to date": [tuition_up_to_date],
        "Scholarship holder": [scholarship_holder],
        "Curricular units 2nd semester (approved)": [
            curricular_units_2nd_sem
        ],
        "Curricular units 1st semester (approved)": [
            curricular_units_1st_sem
        ],
    })

    try:
      prediction = model.predict(input_data)
      prediction_proba = (
          model.predict_proba(input_data)
          if hasattr(model, "predict_proba")
          else None
      )

      st.markdown("---")
      st.subheader("📊 Prediction Results")

      result = prediction[0]

      if result == 1 or str(result).lower() in ["dropout", "1"]:
        st.error(
            "⚠️ **High Risk:** The student is predicted to be at **Risk of"
            " Dropout**."
        )
      else:
        st.success(
            "✅ **Low Risk:** The student is predicted to **Stay Enrolled /"
            " Graduate**."
        )

      if prediction_proba is not None:
        confidence = max(prediction_proba[0]) * 100
        st.info(f"🔍 **Model Confidence:** {confidence:.2f}%")

    except Exception as e:
      st.error(f"❌ Error during prediction: {e}")
