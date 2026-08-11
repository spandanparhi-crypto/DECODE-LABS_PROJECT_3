# 🧠 CUSTOMER SEGMENTATION — UNSUPERVISED LEARNING

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/Machine%20Learning-Unsupervised-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/PCA-Dimensionality%20Reduction-purple?style=for-the-badge" />
<img src="https://img.shields.io/badge/K--Means-Clustering-green?style=for-the-badge" />
<img src="https://img.shields.io/badge/Scikit--Learn-ML-red?style=for-the-badge&logo=scikit-learn" />
<img src="https://img.shields.io/badge/Pandas-Data%20Analysis-blue?style=for-the-badge&logo=pandas" />
<img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-lightblue?style=for-the-badge&logo=numpy" />
<img src="https://img.shields.io/badge/Matplotlib-Visualization-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/Seaborn-Visualization-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />

</p>

<h3 align="center">
AI-Powered Customer Segmentation using PCA, K-Means, Elbow Method & Silhouette Score
</h3>

---

# 📌 PROJECT OVERVIEW

This project implements an end-to-end **Unsupervised Machine Learning pipeline** for discovering hidden customer groups from unlabeled retail data.

The objective is to identify mathematically meaningful customer segments based on customer behavior, purchasing patterns, spending characteristics, and other available features.

The project combines:

- Principal Component Analysis (PCA)
- K-Means Clustering
- Euclidean Distance
- Elbow Method
- Silhouette Score
- Customer Cluster Profiling
- Business Persona Creation
- Business Intelligence Translation

The final output converts mathematical clusters into understandable and actionable customer personas.

---

# 🎯 PROJECT GOAL

The primary goal is to discover hidden patterns in retail customer data without using predefined labels.

The project follows this pipeline:

```text
Raw Retail Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Feature Scaling
        │
        ▼
PCA Dimensionality Reduction
20+ Features → 2/3 Components
        │
        ▼
K-Means Clustering
        │
        ├───────────────┐
        ▼               ▼
Elbow Method      Silhouette Score
        │               │
        └───────┬───────┘
                ▼
       Optimal K Selection
                │
                ▼
       Customer Clustering
                │
                ▼
       Cluster Profiling
                │
                ▼
       Customer Personas
                │
                ▼
   Business Recommendations

# 🌐 STREAMLIT DASHBOARD

## 🚀 Live Local Dashboard

[http://localhost:8501/](http://localhost:8501/)

Run the Streamlit application:

```bash
streamlit run app.py
