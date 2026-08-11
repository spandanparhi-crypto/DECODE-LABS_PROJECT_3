# 🧠 Customer Segmentation — Unsupervised Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Unsupervised-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PCA-Dimensionality%20Reduction-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/K--Means-Clustering-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-red?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

<p align="center">
  <b>AI-Powered Customer Segmentation using PCA, K-Means, Elbow Method & Silhouette Analysis</b>
</p>

---

## 📌 Project Overview

This project implements an **Unsupervised Machine Learning pipeline** to discover hidden customer groups from unlabeled retail data.

The goal is to identify mathematically meaningful customer segments using:

- Principal Component Analysis (PCA)
- K-Means Clustering
- Euclidean Distance
- Elbow Method
- Silhouette Score
- Business Persona Translation

Instead of relying on predefined customer labels, the system automatically discovers groups based on similarities in customer behavior and characteristics.

---

## 🎯 Project Goal

The primary objective is to transform complex, high-dimensional retail customer data into meaningful business segments.

### Pipeline

```text
Raw Retail Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Data Scaling
        │
        ▼
PCA
20+ Features → 2/3 Components
        │
        ▼
K-Means Clustering
        │
        ├── Elbow Method
        │
        └── Silhouette Score
        │
        ▼
Optimal Number of Clusters
        │
        ▼
Customer Personas
        │
        ▼
Business Recommendations
