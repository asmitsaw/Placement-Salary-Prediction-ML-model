# 🎓 Placement Predictor AI

A comprehensive, end-to-end Machine Learning web application designed to predict a student's placement probability and their expected starting salary. The project leverages academic, demographic, skill-based, and lifestyle metrics to provide highly accurate predictions through a modern, responsive **Streamlit** Web Interface.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Dataset Description](#-dataset-description)
4. [Technology Stack](#-technology-stack)
5. [Machine Learning Workflow](#-machine-learning-workflow)
6. [Application & UI Architecture](#-application--ui-architecture)
7. [Installation & Setup](#-installation--setup)
8. [Project Structure](#-project-structure)
9. [Future Enhancements](#-future-enhancements)

---

## 🎯 Project Overview
This project acts as an AI-powered career counselor. By evaluating a student's profile across 20+ different dimensions, it models two target variables:
1. **Placement Probability (Classification)**: Will the student get placed? (Yes/No)
2. **Salary Package (Regression)**: If placed, what is the expected salary? (in LPA - Lakhs Per Annum)

---

## 🧠 Problem Statement
Engineering students often struggle to understand which aspects of their profile need the most improvement to secure good campus placements. This AI agent takes subjective and measurable data points to give empirical placement outcomes and financial expectations, thus helping students pinpoint areas to improve (like backlogs, coding skills, or internships).

---

## 📊 Dataset Description
The model is trained on custom datasets covering 5,000 engineering student records. It has been merged from two separate CSV files:

### Data Sources
*   `indian_engineering_student_placement.csv` (Features)
*   `placement_targets.csv` (Target Variables)

### Feature Categories
*   **Academic Data**: CGPA, 10th %, 12th %, Branch, Active Backlogs, Attendance %.
*   **Skill Ratings (1-10)**: Coding, Communication, Aptitude.
*   **Activities & Achievements**: Projects Completed, Internships, Hackathons, Certifications.
*   **Lifestyle & Environment**: Study Hours/Day, Sleep Hours/Day, Stress Level, Part-time Job, Family Income Level, City Tier, Internet Access, Extracurricular Involvement.

### Target Variables
*   `placement_status`: Binary (1 = Placed, 0 = Not Placed).
*   `salary_lpa`: Continous variable predicting Expected LPA (Exponentially transformed to handle skewness).

---

## 💻 Technology Stack
*   **Data Processing & EDA**: `Pandas`, `NumPy`, `Plotly`, `Matplotlib`, `Seaborn`.
*   **Machine Learning Modeling**: `Scikit-Learn` (Random Forest, Logistic Regression, Decision Trees, SVR).
*   **Model Deployment & UI**: `Streamlit`.
*   **Model Persistence**: `Joblib`.
*   **Styling**: Custom CSS-in-Streamlit with Modern UI, Glassmorphism, Google Fonts (`Inter`).

---

## ⚙️ Machine Learning Workflow
The entire ML workflow is documented in `placement_model.ipynb`:

1.  **Data Ingestion & Merging**: Merging features with target labels on `Student_ID`.
2.  **Data Preprocessing**:
    *   Imputing missing values (`extracurricular_involvement`).
    *   Categorical Encoding using `LabelEncoder` (for Gender, Branch, Income Level, etc.).
    *   Feature Scaling using `StandardScaler` to bring all numerical distributions to mean=0, std=1 (`scaler.pkl`).
3.  **Model Training**:
    *   **Classification** (`placement_model.pkl`): Training classifiers to predict `placement_status`.
    *   **Regression** (`salary_model.pkl`): Training regressors using `np.log1p()` transformed salary data to handle right-skewed salary outliers. The app automatically inverses this using `np.expm1()` during prediction.
4.  **Evaluation**: Models are evaluated on robust metrics (`Accuracy`, `F1-Score` for classification, `RMSE`, `R2 Score` for regression).

---

## 🎨 Application & UI Architecture
The presentation layer is built in `app.py` utilizing Streamlit, but heavily modified using `.streamlit/config.toml` and custom injected CSS for a premium Web Application feel.

### Key UI Features:
*   **Modern Light Theme**: Indigo (`#4f46e5`) and Periwinkle color palette to maintain high visual contrast.
*   **Card Layouts & Modifiers**: Grouping questions into *Academic Profile* and *Activities & Lifestyle*.
*   **UX Enhancements**: Custom slider tracks, thumb animations, and glassmorphic result cards.
*   **Smart Analytics Tip**: Along with the numerical prediction, the UI yields text-based insights (e.g. *"Focus areas: Clear active backlogs, improve CGPA, complete at least one internship..."*).

---

## 🚀 Installation & Setup

### 1. Prerequisites
*   Python 3.9+
*   pip package manager

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```
*The app will be accessible at `http://localhost:8501/`*

---

## 🗂 Project Structure
```text
📦 Salary Projection AI
┣ 📜 app.py                                   # Streamlit Web App Interface
┣ 📜 placement_model.ipynb                    # Data Analysis & Model Training Notebook
┣ 📜 indian_engineering_student_placement.csv # Input Dataset (Features)
┣ 📜 placement_targets.csv                    # Target Dataset (Labels)
┣ 📜 placement_model.pkl                      # Trained Classification Model
┣ 📜 salary_model.pkl                         # Trained Regression Model
┣ 📜 scaler.pkl                               # Scikit-learn StandardScaler
┣ 📜 requirements.txt                         # Dependency List
┗ 📂 .streamlit
  ┗ 📜 config.toml                            # UI Theming Configurations
```

---

## ✨ Future Enhancements
*   **Model Drift Monitoring**: Implement mechanisms to continually retrain models on new graduating batches to prevent data staleness.
*   **Feature Importance Visualization**: Add an interactive chart natively in the UI to show students *why* they got a specific prediction.
*   **API-based Architecture**: Shift `.pkl` loading from the frontend to a dedicated `FastAPI` backend for enhanced scalability and mobile-app integration.

---
*Developed as an AI-powered Analytics Project for Education & Career Placement.*
