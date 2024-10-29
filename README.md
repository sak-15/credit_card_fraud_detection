# Credit Card Fraud Detection 💳

![image](https://github.com/user-attachments/assets/e32d2547-1339-4c76-9748-b89231a10c27)


## Overview
This project aims to build a machine learning model to detect fraudulent transactions in a credit card dataset. The dataset used is highly imbalanced, where a very small fraction of transactions are fraudulent. The goal is to develop models that can effectively identify fraud while minimizing false positives and negatives.

### Objectives:
- Handle imbalanced data to improve fraud detection accuracy.
- Evaluate the models using metrics suited for imbalanced datasets like Precision, Recall, and ROC-AUC.

---

### Know Your Data
The dataset is sourced from [Kaggle's Credit Card Fraud Detection dataset](https://www.kaggle.com/competitions/nus-fintech-recruitment). 

#### Files
- train.csv - the training set. Contains credit card transactions that occurred from Aug - Dec 2021 
- test.csv - the test set for submission. Contains credit card transactions that occurred from Jan - Apr 2022. Only predictions for transactions that occurred in Apr 2022 need to be submitted. Please sort your submission by the transaction ID in ascending order 
- customer.csv - list of customers and their respective customer IDs. A pair of coordinates (x_customer_id, y_customer_id) is provided to indicate the location of the customer 
- terminal.csv - list of merchants (terminals) and their respective terminal IDs. A pair of coordinates (x_terminal_id, y_terminal_id) is provided to indicate the location of the terminal

**Columns in train.csv**
- TRANSACTION_ID - transaction ID
- TX_DATETIME - date and time of transaction
- CUSTOMER_ID - customer ID involved in the transaction
- TERMINAL_ID - terminal ID where transaction occurred
- TX_AMOUNT - amount transacted
- TX_FRAUD - indicates if the transaction is fraudulent. 1 for fraudulent and 0 for legitimate

**Columns in customer.csv**
- CUSTOMER_ID - customer ID
- x_customer_id - x-coordinate of the customer
- y_customer_id - y-coordinate of the customer
- mean_amount - mean amount spent by the customer
- std_amount - standard deviation of the amount spent by the customer
- mean_nb_tx_per_day - mean number of transactions made by the customer per day
- available_terminals - terminals where the customer is able to make transactions. We assume that customers can only make transactions at terminals within a radius of 5 units from the location of the customer
- nb_terminals - number of terminals that the customer can make transactions

**Columns in terminal.csv**
- TERMINAL_ID - terminal ID
- x_terminal_id - x-coordinate of the terminal
- y_terminal_id - y-coordinate of the terminal

---

### Data Pre-processing

![image](https://github.com/user-attachments/assets/842229a8-10f8-4e9b-8f37-d3a4ce93a9ed)

Preprocessing is the process of cleaning the dataset. In this step, we will apply different methods to clean the raw data to feed more meaningful data for the modeling phase. This method includes

- Remove duplicates or irrelevant samples
- Update missing values with the most relevant values 
- Convert one data type to another example, categorical to integers, etc.

---

### Handling Imbalance Data

**What is Imbalanced data ?**

Imbalanced data refers to a dataset in which the classes are not represented equally. In classification tasks, it is common to encounter imbalanced data when one class (or outcome) is far more frequent than the others. For example, in a dataset for credit card fraud detection, the number of legitimate transactions (non-fraud) vastly outnumbers fraudulent transactions.

![image](https://github.com/user-attachments/assets/02a00bef-d9fd-4bc2-82e0-54f94c7b161b)

In credit card fraud detection, sampling is very important because of the highly imbalanced dataset—fraud cases are much rarer than legitimate transactions. Without handling this imbalance, most models would get good accuracy by simply predicting all transactions as non-fraud (since the majority of cases are legitimate). This approach, however, would miss most fraudulent transactions, which we want to avoid.

**1. Why Sampling is Important**

Sampling helps create a more balanced dataset by either:

- Increasing the number of fraud cases (oversampling), or
- Reducing the number of non-fraud cases (undersampling).
  
This balance allows the model to learn fraud patterns effectively, improving its ability to detect fraud without being overwhelmed by the majority class (legitimate transactions).

**2. Types of Sampling to Consider**

**a. SMOTE (Synthetic Minority Over-sampling Technique)**
- When to use: If your dataset is large and you want to create more fraud cases by generating synthetic data points.
- How it works: SMOTE creates synthetic fraud cases by interpolating between existing fraud cases, helping avoid duplicating exact entries.
- Pros: Reduces the likelihood of overfitting as it doesn’t duplicate cases but instead generates new ones based on the minority class.

**b. SMOTEENN (Combination of SMOTE and Edited Nearest Neighbors)**
- When to use: When you want a mix of oversampling and undersampling for cleaner data.
- How it works: SMOTE generates synthetic fraud cases, and then Edited Nearest Neighbors removes overlapping points from the majority class that might confuse the model.
- Pros: Creates a more refined dataset by keeping only clearly separated legitimate and fraudulent cases.

![image](https://github.com/user-attachments/assets/d96e021d-5cfb-44c6-abfa-627f8d6108ca)

---

### Metrics for imbalanced data

Selecting the right metrics is crucial since you’re dealing with a highly imbalanced dataset where fraudulent transactions are rare compared to legitimate ones.

**Key Metrics to Focus On:**

![image](https://github.com/user-attachments/assets/0901c35b-1e7f-4ab1-b6b9-ca924c061748)

- **Recall (Sensitivity):** Since missing fraudulent transactions (false negatives) is often more costly than a few extra checks on legitimate ones, recall is crucial. High recall ensures that most fraudulent transactions are detected, even if it means occasionally flagging some legitimate transactions.

- **Precision:** Precision is important if you want to minimize the number of false positives (incorrectly flagged legitimate transactions). High precision ensures that flagged transactions are indeed likely to be fraudulent, avoiding unnecessary friction for legitimate users.

- **F1 Score:** Since both precision and recall are relevant, the F1 score serves as a balanced metric for fraud detection, giving you a single indicator of the model’s performance by balancing precision and recall.

- **Precision-Recall AUC:** Given the class imbalance, Precision-Recall AUC is often more insightful than ROC-AUC. It will help you see how well your model performs across different thresholds specifically for detecting fraud.

---
### Hyperparameter tuning 

Hyperparameter tuning for an XGBoost model involves adjusting the various parameters that control the behavior of the model to find the best combination that optimizes performance. XGBoost has many hyperparameters, but some common ones for tuning include:

<img width="716" alt="image" src="https://github.com/user-attachments/assets/a14b41d8-1f0a-442c-bfd3-4f7d016efbd6">



**Why Hyperparameter Tuning is Important ?**

When building a model for tasks like credit card fraud detection, having optimized hyperparameters is essential because:

- **Class Imbalance:** Fraud cases are rare, so the model must be tuned to recognize patterns in these rare cases without overfitting.
- **Performance Needs:** For fraud detection, precision and recall are often more important than accuracy. Tuning hyperparameters helps the model focus on identifying fraud cases correctly, balancing true positives (fraud) and false positives (non-fraud).
- **Avoiding Overfitting:** Without tuning, the model might learn details of the training data too well, leading to high performance in training but poor generalization to new data.


**Why Random Search Fits Our Case ?**

In credit card fraud detection with XGBoost, Random Search is ideal because:

- **Large Parameter Space:** XGBoost has many hyperparameters with wide ranges. Testing every combination with Grid Search would be extremely time-consuming.
- **Time Constraints:** Fraud detection requires quick adaptation; a faster tuning method allows more frequent model updates.
- **Effective Exploration:** Random Search can cover the most impactful parts of the parameter space efficiently, often finding combinations close to optimal with fewer trials.
  
![image](https://github.com/user-attachments/assets/33a8288d-9573-453b-a066-eaf8c4e89228)

---

### Project Structure (Planned)

```bash
├── data/                   # Dataset files
├── notebooks/              # Jupyter notebooks for exploration and modeling
├── requirements.txt/       # Required libraries for setting up environment
├── hyperparameters/        # estimators for xgb
└── README.md               # Project documentation

