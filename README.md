# 🎓 EduPredict AI: Institutional Retention & Dropout Risk Analyzer

An enterprise-grade, full-stack machine learning web application designed to predict student dropout risks in higher education. It features a modern interactive dashboard built with **Streamlit**, a robust API architecture powered by **FastAPI** and **Pydantic**, and a heavy production-ready inference pipeline using **Scikit-Learn**, **Pandas**, **Joblib**, and **ColumnTransformer**.

---

## 🚀 Live App
Check out the fully deployed live application here:  
👉 [EduPredict AI Streamlit Cloud App](https://student-dropout-pred-ml-app-mnfxzdc9xbelmihm6esrnh.streamlit.app/)

---

## 🛠️ Key Technical Features & Tech Stack

* **Frontend Dashboard (`Streamlit`)**: 
  * Responsive, multi-tabbed interactive UI configured with custom CSS styles (`#0f172a` dark theme layout).
  * Organized across four primary decision dimensions: Background, 1st Semester Records, 2nd Semester Records, and Macro-Economics.
* **Backend Architecture (`FastAPI` & `Pydantic`)**:
  * High-performance asynchronous API endpoints (`/predict`) designed to handle rigorous data payload validation and schema checking through Pydantic data models.
* **Machine Learning Pipeline (`Scikit-Learn` & `Joblib`)**:
  * Employs advanced preprocessing pipelines built with **ColumnTransformer** and **Pipeline** structures to efficiently handle mixed data types (numerical and categorical scaling/encoding).
  * Trained utilizing classification algorithms and serialized via `joblib` into a high-performance production artifact (`student_dropout_model.pkl`).
* **Data Processing & Visualization (`Pandas` & `Matplotlib`)**:
  * Dynamic data wrangling, feature engineering, and statistical analytics powered under the hood by `pandas`.
* **Cloud Deployment (`Streamlit Community Cloud`)**:
  * Seamless continuous deployment and cloud hosting integrated directly with GitHub source control.

---

## 📂 Project Directory Structure

```text
student-dropout-pred-ml-app/
│
├── app.py                      # Main Streamlit frontend & direct ML inference interface
├── backend.py                  # FastAPI server & Pydantic request data validation schemas
├── student_dropout_model.pkl   # Serialized Scikit-Learn pipeline model with ColumnTransformer
├── requirements.txt            # Project Python dependencies list
└── README.md                   # Comprehensive project documentation
