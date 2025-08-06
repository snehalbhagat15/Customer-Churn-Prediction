#  Customer Churn Prediction Using Machine Learning

This project aims to predict whether a customer will churn (leave a service) using supervised machine learning techniques. We used a telecom customer dataset and applied data preprocessing, exploratory analysis, and classification modeling to predict customer churn effectively.

---

## Problem Statement

Customer churn is a major problem in many industries. By identifying customers who are likely to leave, businesses can take proactive steps to retain them and reduce revenue loss.

---

##  Tools & Libraries Used

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Jupyter Notebook
- Machine Learning Algorithm: Random Forest Classifier

---

##  Model Evaluation

- **Accuracy:** 78.99%
- **Classification Report:**
  - Precision (Class 0 - No churn): 0.82
  - Precision (Class 1 - Churn): 0.65
  - F1-score (macro average): 0.70

- **Confusion Matrix:**

[[942 94]
[202 171]]

---

##  Key Highlights

- Handled missing values and categorical features
- Applied label encoding and feature scaling
- Evaluated model performance using accuracy, precision, recall, and F1-score
- Visualized feature importance for business insights

---

##  Future Improvements

- Try advanced models like XGBoost or LightGBM
- Hyperparameter tuning using GridSearchCV
- Handle class imbalance with SMOTE or class weights

---

##  How to Run

1. Clone the repo or download the notebook
2. Install dependencies
3. Open `churn_prediction.ipynb` in Jupyter
4. Run cells sequentially

---


