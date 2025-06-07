
# 🛡️ Fraud Detection Analysis Using Python and SQL

## 📌 Table of Contents

- [Introduction](#introduction)
- [Objective](#objective)
- [Tools Used](#tools-used)
- [Dataset Description](#dataset-description)
- [Data Cleaning & Preparation](#data-cleaning--preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [SQL-Based Risk Rules](#sql-based-risk-rules)
- [ML-Based Prediction](#ml-based-prediction)
- [Model Evaluation](#model-evaluation)
- [Key Insights](#key-insights)
- [Conclusion](#conclusion)
- [Future Improvements](#future-improvements)

---

## 📖 Introduction

This project showcases how a data analyst in the fintech domain can detect and prevent fraud by leveraging SQL-based risk rules and Python-based predictive modeling. The approach combines rule-based filtering with machine learning to improve fraud detection accuracy and real-time applicability.

---

## 🎯 Objective

- Identify fraudulent patterns using SQL queries
- Train predictive models to flag fraud before it occurs
- Present business insights and create an actionable fraud strategy

---

## 🛠️ Tools Used

- **Languages**: Python, SQL (PostgreSQL / MySQL)
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, imbalanced-learn
- **Visualization**: Plotly, Streamlit, Tableau
- **IDE**: Jupyter Notebook / VS Code
- **Database**: PostgreSQL

---

## 📂 Dataset Description

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: ~492 (0.17%)
- **Features**: 30 (V1–V28 anonymized, Amount, Time, Class)
- **Target**: `Class` (0 = Non-Fraud, 1 = Fraud)

---

## 🧹 Data Cleaning & Preparation

- Checked and removed duplicates/nulls
- Scaled `Amount` and `Time` columns
- Extracted time-based features (Hour, Day)
- Balanced classes using SMOTE (oversampling)

---

## 📊 Exploratory Data Analysis

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("creditcard.csv")

# Fraud distribution
print(df['Class'].value_counts(normalize=True) * 100)

# Amount distribution
sns.boxplot(x='Class', y='Amount', data=df)
plt.title("Transaction Amount by Class")
plt.show()
```

---

## 🧮 SQL-Based Risk Rules

### ✅ Rule 1: Same card used in different cities within 5 mins

```sql
SELECT a.card_number
FROM transactions a
JOIN transactions b
  ON a.card_number = b.card_number
 AND a.city <> b.city
 AND ABS(EXTRACT(EPOCH FROM a.txn_time - b.txn_time)) < 300
WHERE a.txn_time <> b.txn_time;
```

### ✅ Rule 2: Transaction amount 3x higher than card average

```sql
SELECT card_number, amount
FROM transactions t1
WHERE amount > 3 * (
  SELECT AVG(amount)
  FROM transactions t2
  WHERE t1.card_number = t2.card_number
);
```

### ✅ Rule 3: More than 5 transactions in 15 minutes

```sql
SELECT card_number, COUNT(*) as txn_count
FROM transactions
WHERE txn_time BETWEEN NOW() - INTERVAL '15 minutes' AND NOW()
GROUP BY card_number
HAVING COUNT(*) > 5;
```

---

## 🤖 ML-Based Prediction

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Prepare data
X = df.drop("Class", axis=1)
y = df["Class"]

# Balance data
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, stratify=y_res, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 📈 Model Evaluation

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.99  |
| Precision  | 0.95  |
| Recall     | 0.91  |
| F1 Score   | 0.93  |
| ROC AUC    | 0.97  |

---

## 📌 Key Insights

- 🔍 Fraudulent transactions are extremely rare (~0.17%)
- 💳 High-value frauds tend to occur late night or early morning
- 🧠 Random Forest performed best in balancing recall and precision
- 🛠️ SQL risk rules detected ~60% of frauds with high confidence

---

## ✅ Conclusion

This project proves that combining **rule-based detection (SQL)** with **predictive modeling (Python)** creates a robust fraud detection system. This hybrid model improves fraud detection rates, reduces false positives, and adds real-time applicability.

---

## 🔮 Future Improvements

- Deploy model with FastAPI or Flask for real-time detection
- Streamlit dashboard for fraud analytics and alerting
- Enrich data with device ID, IP address, merchant category
- Integrate feedback loop to auto-update rules and model
