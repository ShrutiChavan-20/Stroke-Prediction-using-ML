# 🧠 Stroke Prediction using Machine Learning  

## 📌 Project Overview  
This project focuses on predicting the likelihood of a patient having a **stroke** based on healthcare data.  
The dataset used is the **Stroke Prediction Dataset** (from Kaggle), which contains patient information such as age, hypertension, heart disease, glucose level, BMI, and lifestyle factors.  

The project builds an **end-to-end machine learning pipeline** covering data preprocessing, model training, evaluation, and performance comparison across multiple algorithms.  

---

## 📂 Dataset  
- **Name:** healthcare-dataset-stroke-data.csv  
- **Features include:**  
  - `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`  
  - **Target:** `stroke` (0 = No Stroke, 1 = Stroke)  

📊 Dataset link: [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)  

---

## ⚙️ Workflow  
1. **Data Preprocessing**  
   - Handling missing values (e.g., BMI)  
   - Encoding categorical variables (OneHot & Label Encoding)  
   - Feature scaling for numerical columns  

2. **Exploratory Data Analysis (EDA)**  
   - Distribution plots for risk factors  
   - Correlation heatmap between features  
   - Class imbalance analysis  

3. **Model Building**  
   Implemented and compared the following models:  
   - Logistic Regression  
   - Decision Tree  
   - K-Nearest Neighbors (KNN)  
   - Support Vector Machine (SVM)  
   - XGBoost  

4. **Model Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC  
   - Visualization: Confusion Matrix, ROC Curve  

---

## 📊 Results  
- **Best Model:** XGBoost  
- **Performance Highlights:**  
  - Achieved higher classification accuracy compared to traditional ML models  
  - Balanced tradeoff between Precision and Recall (important for healthcare applications)  

---

## 🚀 Technologies Used  
- **Programming:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost  
- **Tools:** Jupyter Notebook  

---

## 📌 Key Insights  
- Age, hypertension, heart disease, and glucose levels are strong predictors of stroke.  
- XGBoost provided the most reliable predictions, outperforming Logistic Regression and Decision Trees.  
- Proper handling of class imbalance and medical interpretability is crucial in healthcare ML projects.  

---
