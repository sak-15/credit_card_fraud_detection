# Credit Card Fraud Detection 💳

## Overview
This project aims to build a machine learning model to detect fraudulent transactions in a credit card dataset. The dataset used is highly imbalanced, where a very small fraction of transactions are fraudulent. The goal is to develop models that can effectively identify fraud while minimizing false positives and negatives.

### Objectives:
- Handle imbalanced data to improve fraud detection accuracy.
- Experiment with both supervised models and unsupervised anomaly detection methods.
- Evaluate the models using metrics suited for imbalanced datasets like Precision, Recall, and ROC-AUC.

---

## Dataset
The dataset is sourced from [Kaggle's Credit Card Fraud Detection dataset](https://www.kaggle.com/competitions/nus-fintech-recruitment). 

#### Files
- train.csv - the training set. Contains credit card transactions that occurred from Aug - Dec 2021 
- test.csv - the test set for submission. Contains credit card transactions that occurred from Jan - Apr 2022. Only predictions for transactions that occurred in Apr 2022 need to be submitted. Please sort your submission by the transaction ID in ascending order 
- customer.csv - list of customers and their respective customer IDs. A pair of coordinates (x_customer_id, y_customer_id) is provided to indicate the location of the customer 
- terminal.csv - list of merchants (terminals) and their respective terminal IDs. A pair of coordinates (x_terminal_id, y_terminal_id) is provided to indicate the location of the terminal 

---

## Project Structure (Planned)

```bash
├── data/                   # Dataset files
├── notebooks/              # Jupyter notebooks for exploration and modeling
├── requirements.txt/       # Required libraries for setting up environment
├── hyperparameters/        # estimators for xgb
└── README.md               # Project documentation
