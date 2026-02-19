# 🛡 Fraud Transaction Detection System

An end-to-end Machine Learning pipeline for detecting fraudulent financial transactions using large-scale, highly imbalanced data (6.3M+ records).  
The system leverages advanced feature engineering, ensemble learning (XGBoost), time-based validation, and SHAP explainability to deliver high-confidence fraud predictions aligned with real-world financial risk patterns.

---

## 🚀 Project Objective

To proactively detect fraudulent transactions in a financial system by:

- Handling extreme class imbalance (~0.13% fraud rate)
- Preventing data leakage using time-aware validation
- Capturing behavioral fraud patterns via engineered financial signals
- Ensuring interpretability through SHAP explainability
- Designing a deployment-ready fraud scoring framework

---

## 📊 Dataset Overview

- **Total Records:** 6,362,620 transactions  
- **Time Span:** 744 hourly steps (~30 days simulation)  
- **Fraud Rate:** ~0.13% (Highly imbalanced dataset)  
- **Transaction Types:** CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER  

The dataset simulates fraudulent agents taking control of accounts and attempting to drain funds through transfer and cash-out operations.

---

## 🧠 Key Technical Highlights

### 1️⃣ Feature Engineering

Designed high-signal behavioral features aligned with fraud mechanics:

- `balanceDiffOrig` – Captures account draining magnitude
- `origBalanceError` – Detects arithmetic inconsistencies
- `isOrigZeroAfter` – Flags full account depletion
- `log_amount` – Reduces skew and stabilizes model training

These features directly model the fraud pattern described in the dataset.

---

### 2️⃣ Data Leakage Prevention

Implemented strict **time-based train-test split**:

- Train → First 80% of transaction steps  
- Test → Final 20% of steps  

This mirrors real-world deployment where models predict future fraud from historical data.

---

### 3️⃣ Model Benchmarking

Compared multiple classification models:

| Model | Purpose |
|-------|---------|
| Logistic Regression | Linear baseline |
| Random Forest | Non-linear rule-based learning |
| XGBoost | Gradient boosting with error correction |

**Final Model:** XGBoost with imbalance-aware optimization  

