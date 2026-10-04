# 🎓 EduPredict AI: Student Dropout Risk Predictor

An end-to-end machine learning app that predicts whether a student is at risk of dropping out, using academic, demographic, socioeconomic and macroeconomic data. Built with a scikit-learn pipeline, a FastAPI backend and a Streamlit dashboard.

👉 **[Live Demo](https://student-dropout-pred-ml-app-mnfxzdc9xbelmihm6esrnh.streamlit.app/)**

---

## 📸 Screenshots



| 

![Screenshot 1](Screenshot%202026-09-13%20225013.png)

 | 

![Screenshot 2](Screenshot%202026-09-13%20225023.png)

 | 

![Screenshot 3](Screenshot%202026-09-13%20225040.png)

 |

---

## 🎯 Problem

Early identification of at-risk students lets institutions offer timely support. This project predicts dropout risk from student records so that advisors can intervene early.

> **Note:** This tool is meant for early support only, not for denying admission or opportunities to any student.

---

## 📊 Dataset

- **Source:** Kaggle, Student Dropout and Academic Success dataset
- **Size:** 4,424 students, 36 features + 1 target column
- **Feature groups:** demographics, parents' qualification/occupation, scholarship and tuition status, 1st and 2nd semester curricular units (enrolled, evaluated, approved, grades), and macroeconomic indicators (unemployment rate, inflation rate, GDP)
- **Target:** Binary, Dropout = 1, otherwise = 0
- **Split:** 80/20 train/test (3,539 train, 885 test)
- **Class balance (test set):** 569 non-dropout vs 316 dropout (about 64% / 36%)

---

## 🧠 Approach

1. Exploratory data analysis and outlier analysis (`Student_Dropout_Model.ipynb`)
2. Label encoding of the target column
3. Preprocessing with `ColumnTransformer` + `StandardScaler` inside a scikit-learn `Pipeline`
4. Model training and evaluation (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
5. Pipeline saved with `joblib` as `student_dropout_model.pkl`
6. Served through a FastAPI `/predict` endpoint (Pydantic validation) and a Streamlit UI

---

## 📈 Results (test set, 885 students)

| Metric | Score |
|---|---|
| Accuracy | 0.7831 |
| Precision (dropout) | 0.8196 |
| Recall (dropout) | 0.5032 |
| F1-score (dropout) | 0.6235 |
| ROC-AUC | 0.8180 |

**Confusion matrix**

| | Predicted: Not dropout | Predicted: Dropout |
|---|---|---|
| **Actual: Not dropout** | 534 | 35 |
| **Actual: Dropout** | 157 | 159 |

The model is precise (about 82% of students it flags as at-risk really drop out) but conservative: it catches about half of the actual dropouts. Full analysis: see [Report.pdf](Report.pdf).

---

## 🛠️ Tech Stack

- **ML:** Python, Pandas, NumPy, scikit-learn, Joblib
- **Backend:** FastAPI, Pydantic (request validation)
- **Frontend:** Streamlit, Matplotlib
- **Deployment:** Streamlit Community Cloud

---

## 📂 Project Structure

```
.
├── app.py                        # Streamlit UI
├── backend.py                    # FastAPI server + Pydantic schemas
├── Student_Dropout_Model.ipynb   # Training and evaluation notebook
├── student_dropout_model.pkl     # Trained scikit-learn pipeline
├── Report.pdf                    # Project report
├── requirements.txt
└── README.md
```

---

## ⚙️ Run Locally

```
git clone https://github.com/HamzaWaseem2005/Student-Dropout-Pred-ML-app.git
cd Student-Dropout-Pred-ML-app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚠️ Limitations

- Recall for dropout is about 50%, so roughly half of the students who drop out are not flagged. The model should support advisors, not replace them.
- The dataset comes from a single institution, so results may not generalise elsewhere.
- Predictions are probabilities, not certainties.

## 🚀 Future Improvements

- Improve dropout recall with class weights or decision-threshold tuning
- Compare more models (Random Forest, XGBoost) with cross-validation
- Add model explainability (SHAP) to show why a student is flagged
- Add fairness checks across demographic groups
- Containerise with Docker

---

## 👤 Author

**Muhammad Hamza Waseem**
[GitHub](https://github.com/HamzaWaseem2005) · [LinkedIn](https://www.linkedin.com/in/muhammad-hamza-waseem-976535336)

## 📄 License

Apache-2.0
