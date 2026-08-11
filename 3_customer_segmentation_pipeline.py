"""
customer_segmentation.py
-------------------------
Project 3: Unsupervised Learning (Customer Segmentation)

Pipeline:
  1. Load unlabeled retail data (28 numeric features, no target column).
  2. Standardize features (distance-based algorithms require this).
  3. PCA -> reduce to 2D and 3D for visualization + noise reduction.
  4. Elbow Method (inertia) + Silhouette Score across k=2..10 to
     mathematically justify the optimal number of clusters.
  5. Fit final K-Means model at the chosen k.
  6. Profile each cluster's feature means and translate into
     actionable business "Personas".
  7. Export: cluster assignments CSV, persona summary CSV, and all charts.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 10

OUT = "plots"

# =================================================================
# 1. LOAD DATA
# =================================================================
df = pd.read_csv("retail_customers.csv")
customer_ids = df["CustomerID"]
X_raw = df.drop(columns=["CustomerID"])
feature_names = X_raw.columns.tolist()

print(f"Loaded {X_raw.shape[0]} customers, {X_raw.shape[1]} features")
print(f"Missing values: {X_raw.isna().sum().sum()}")

# =================================================================
# 2. STANDARDIZE  (critical: K-Means/PCA are distance-based, and raw
#    features here range from 0-1 rates to $1000s of spend -- without
#    scaling, MonetaryTotal alone would dominate every distance calc)
# =================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# =================================================================
# 3. PCA -> 2D and 3D
# =================================================================
pca_full = PCA().fit(X_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(range(1, len(cum_var) + 1), cum_var, marker="o", color="#2C7FB8")
ax.axhline(0.80, color="gray", linestyle="--", linewidth=1, label="80% variance")
ax.set_xlabel("Number of Principal Components")
ax.set_ylabel("Cumulative Explained Variance")
ax.set_title("PCA: Cumulative Explained Variance by Component")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/1_pca_explained_variance.png")
plt.close(fig)

pca_2d = PCA(n_components=2, random_state=42)
X_pca2 = pca_2d.fit_transform(X_scaled)
pca_3d = PCA(n_components=3, random_state=42)
X_pca3 = pca_3d.fit_transform(X_scaled)

print(f"\n2D PCA explained variance: {pca_2d.explained_variance_ratio_.round(3)} "
      f"(total {pca_2d.explained_variance_ratio_.sum():.1%})")
print(f"3D PCA explained variance: {pca_3d.explained_variance_ratio_.round(3)} "
      f"(total {pca_3d.explained_variance_ratio_.sum():.1%})")

# Top loadings driving PC1 / PC2 -- helps name the axes later
loadings = pd.DataFrame(
    pca_2d.components_.T, index=feature_names, columns=["PC1", "PC2"]
)
top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(6)
top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(6)
print("\nTop features driving PC1:\n", loadings.loc[top_pc1.index, "PC1"])
print("\nTop features driving PC2:\n", loadings.loc[top_pc2.index, "PC2"])

# =================================================================
# 4. ELBOW METHOD + SILHOUETTE SCORE  (k = 2..10)
# =================================================================
k_range = range(2, 11)
inertias = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_k = km.fit_predict(X_pca2)  # cluster on PCA space (denoised, 2D)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_pca2, labels_k))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(list(k_range), inertias, marker="o", color="#D95F02")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)")
axes[0].set_title("Elbow Method")
axes[0].set_xticks(list(k_range))

axes[1].plot(list(k_range), sil_scores, marker="o", color="#1B9E77")
best_k_idx = int(np.argmax(sil_scores))
best_k = list(k_range)[best_k_idx]
axes[1].scatter([best_k], [sil_scores[best_k_idx]], color="red", zorder=5,
                s=90, label=f"Best k={best_k}")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Average Silhouette Score")
axes[1].set_title("Silhouette Score by k")
axes[1].set_xticks(list(k_range))
axes[1].legend()

fig.tight_layout()
fig.savefig(f"{OUT}/2_elbow_and_silhouette.png")
plt.close(fig)

print(f"\nInertia by k: {dict(zip(k_range, np.round(inertias, 1)))}")
print(f"Silhouette by k: {dict(zip(k_range, np.round(sil_scores, 3)))}")
print(f"\n==> Optimal k selected by max silhouette score: k={best_k}")

# =================================================================
# 5. FINAL K-MEANS MODEL at chosen k
# =================================================================
FINAL_K = best_k
kmeans_final = KMeans(n_clusters=FINAL_K, n_init=25, random_state=42)
cluster_labels = kmeans_final.fit_predict(X_pca2)
final_silhouette = silhouette_score(X_pca2, cluster_labels)
sample_sil = silhouette_samples(X_pca2, cluster_labels)

print(f"\nFinal model: k={FINAL_K}, overall silhouette score = {final_silhouette:.3f}")

df["Cluster"] = cluster_labels
df["PC1"] = X_pca2[:, 0]
df["PC2"] = X_pca2[:, 1]
df["PC3"] = X_pca3[:, 2] if X_pca3.shape[1] > 2 else np.nan
df["SilhouetteSample"] = sample_sil

# --- Silhouette diagram (per-cluster quality) ---
fig, ax = plt.subplots(figsize=(7, 5))
y_lower = 10
colors = plt.cm.tab10(np.linspace(0, 1, FINAL_K))
for i in range(FINAL_K):
    vals = np.sort(sample_sil[cluster_labels == i])
    size = vals.shape[0]
    y_upper = y_lower + size
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, color=colors[i], alpha=0.8)
    ax.text(-0.05, y_lower + size / 2, str(i))
    y_lower = y_upper + 10
ax.axvline(final_silhouette, color="red", linestyle="--", label=f"Avg = {final_silhouette:.3f}")
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster")
ax.set_title(f"Silhouette Plot (k={FINAL_K})")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/3_silhouette_diagram.png")
plt.close(fig)

# --- 2D cluster scatter in PCA space ---
fig, ax = plt.subplots(figsize=(7.5, 6))
for i in range(FINAL_K):
    mask = cluster_labels == i
    ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1], s=18, alpha=0.6,
               color=colors[i], label=f"Cluster {i} (n={mask.sum()})")
centers = kmeans_final.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=250, c="black",
           edgecolor="white", linewidth=1.5, label="Centroids", zorder=5)
ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title(f"Customer Segments in PCA Space (K-Means, k={FINAL_K})")
ax.legend(loc="best", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/4_clusters_pca_2d.png")
plt.close(fig)

# --- 3D cluster scatter ---
fig = plt.figure(figsize=(8, 6.5))
ax = fig.add_subplot(111, projection="3d")
for i in range(FINAL_K):
    mask = cluster_labels == i
    ax.scatter(X_pca3[mask, 0], X_pca3[mask, 1], X_pca3[mask, 2],
               s=14, alpha=0.6, color=colors[i], label=f"Cluster {i}")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title(f"Customer Segments in 3D PCA Space (k={FINAL_K})")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/5_clusters_pca_3d.png")
plt.close(fig)

# =================================================================
# 6. CLUSTER PROFILING -> business personas
# =================================================================
profile = df.groupby("Cluster")[feature_names].mean().round(2)
profile["ClusterSize"] = df["Cluster"].value_counts().sort_index()
profile["PctOfCustomers"] = (profile["ClusterSize"] / len(df) * 100).round(1)

# z-scored profile (vs overall population) to see what's distinctly
# HIGH or LOW per cluster -- this is what actually drives persona naming
pop_mean = X_raw.mean()
pop_std = X_raw.std()
z_profile = (profile[feature_names] - pop_mean) / pop_std

profile.to_csv("cluster_profile_raw_means.csv")
z_profile.round(2).to_csv("cluster_profile_zscores.csv")

# Heatmap of standardized cluster profiles
fig, ax = plt.subplots(figsize=(11, max(4, FINAL_K * 0.9)))
im = ax.imshow(z_profile.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
ax.set_yticks(range(FINAL_K))
ax.set_yticklabels([f"Cluster {i}" for i in range(FINAL_K)])
ax.set_title("Cluster Profiles (Z-score vs. overall customer base)")
fig.colorbar(im, ax=ax, shrink=0.7, label="Std. deviations from mean")
fig.tight_layout()
fig.savefig(f"{OUT}/6_cluster_profile_heatmap.png")
plt.close(fig)

# Cluster sizes bar chart
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([f"C{i}" for i in range(FINAL_K)], profile["ClusterSize"], color=colors)
for i, v in enumerate(profile["ClusterSize"]):
    ax.text(i, v + 5, str(int(v)), ha="center", fontsize=9)
ax.set_ylabel("Number of Customers")
ax.set_title("Cluster Sizes")
fig.tight_layout()
fig.savefig(f"{OUT}/7_cluster_sizes.png")
plt.close(fig)

print("\n=== Cluster Profile (raw feature means) ===")
print(profile[["ClusterSize", "PctOfCustomers", "MonetaryTotal", "Frequency_Purchases",
                "Recency_Days", "AnnualIncome_k", "DiscountUsageRate", "AppSessionMinPerMonth"]])

df.to_csv("customers_with_clusters.csv", index=False)

print("\nAll artifacts saved: customers_with_clusters.csv, cluster_profile_raw_means.csv,")
print("cluster_profile_zscores.csv, and 7 charts in ./plots/")
