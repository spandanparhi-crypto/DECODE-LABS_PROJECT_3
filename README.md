# 🧠 CUSTOMER SEGMENTATION — UNSUPERVISED LEARNING

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Unsupervised-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PCA-Dimensionality%20Reduction-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/K--Means-Clustering-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-red?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-blue?style=for-the-badge&logo=pandas" />
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-lightblue?style=for-the-badge&logo=numpy" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

<p align="center">
  <b>AI-Powered Customer Segmentation using PCA, K-Means, Elbow Method & Silhouette Analysis</b>
</p>

---

# 📌 PROJECT OVERVIEW

**Customer Segmentation — Unsupervised Learning** is an end-to-end Machine Learning project designed to discover hidden customer groups from unlabeled retail data.

The project uses mathematical distance-based algorithms to identify customers with similar purchasing behavior, transaction patterns, and customer characteristics.

The complete pipeline includes:

- Data Loading
- Data Cleaning
- Missing Value Handling
- Duplicate Removal
- Exploratory Data Analysis
- Feature Engineering
- Feature Selection
- Feature Scaling
- Principal Component Analysis
- Elbow Method
- Silhouette Score
- K-Means Clustering
- Cluster Profiling
- Customer Persona Generation
- Business Recommendations
- Data Visualization
- Automated Report Generation
- Streamlit Dashboard

---

# 🎯 PROJECT GOAL

The primary objective is to transform complex, high-dimensional retail customer data into meaningful customer segments.

The project discovers hidden mathematical groupings without using predefined target labels.

### Complete Pipeline

```text
Raw Retail Dataset
        │
        ▼
Data Loading
        │
        ▼
Data Cleaning
        │
        ▼
Missing Value Handling
        │
        ▼
Duplicate Removal
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Feature Selection
        │
        ▼
Feature Scaling
        │
        ▼
PCA
20+ Features → 2/3 Components
        │
        ▼
Elbow Method
        │
        ▼
Silhouette Score
        │
        ▼
Optimal K
        │
        ▼
K-Means Clustering
        │
        ▼
Cluster Profiling
        │
        ▼
Customer Personas
        │
        ▼
Business Recommendations
        │
        ▼
Streamlit Dashboard

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)
