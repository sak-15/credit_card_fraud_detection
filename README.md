# Credit Card Fraud Detection

## Overview
This project aims to build a machine learning model to detect fraudulent transactions in a credit card dataset. The dataset used is highly imbalanced, where a very small fraction of transactions are fraudulent. The goal is to develop models that can effectively identify fraud while minimizing false positives and negatives.

### Objectives:
- Handle imbalanced data to improve fraud detection accuracy.
- Experiment with both supervised models and unsupervised anomaly detection methods.
- Evaluate the models using metrics suited for imbalanced datasets like Precision, Recall, and ROC-AUC.

---

## Dataset
The dataset is sourced from [Kaggle's Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains 284,807 transactions, with 492 labeled as fraudulent (about 0.17% of the total data). Features have been transformed using PCA for privacy reasons.

---

## Project Structure (Planned)

```bash
├── data/                   # Dataset files
├── notebooks/              # Jupyter notebooks for exploration and modeling
├── src/                    # Scripts for data preprocessing and model training
└── README.md               # Project documentation
