# 🍷 Wine Clustering & Dimensionality Reduction using K-Means and PCA

## 📌 Project Overview

This project implements an **Unsupervised Machine Learning** workflow to analyze and group chemical profiles of wines without ground-truth labels. 

The analysis involves:
- Feature distribution analysis and correlation mapping.
- Feature standardization for distance-based algorithms.
- Determining optimal cluster count ($K$) using the **Elbow Method** (Inertia) and **Silhouette Analysis**.
- Clustering comparison between **Standard K-Means** and **Bisecting K-Means**.
- 2D projection and cluster boundary visualization using **Principal Component Analysis (PCA)**.

---

## 📊 Dataset

The project utilizes the **Wine Recognition Dataset** from `sklearn.datasets.load_wine`.

- **Total Samples:** 178 wine instances
- **Missing Values:** 0
- **Total Features:** 13 continuous chemical constituents

### Features

| # | Feature | Description |
| :---: | :--- | :--- |
| **1** | `alcohol` | Alcohol content |
| **2** | `malic_acid` | Malic acid level |
| **3** | `ash` | Ash content |
| **4** | `alcalinity_of_ash` | Alkalinity of ash |
| **5** | `magnesium` | Magnesium level |
| **6** | `total_phenols` | Total phenolic compounds |
| **7** | `flavanoids` | Flavanoid content |
| **8** | `nonflavanoid_phenols` | Non-flavanoid phenols |
| **9** | `proanthocyanins` | Proanthocyanins level |
| **10** | `color_intensity` | Color intensity measure |
| **11** | `hue` | Wine hue |
| **12** | `od280/od315_of_diluted_wines` | Optical density ratio |
| **13** | `proline` | Proline amino acid concentration |

---

## ⚙️ Exploratory Data Analysis & Preprocessing

1. **Distribution Inspection:** Plotted 15-bin histograms across all 13 features to examine skewness and spread.
2. **Correlation Heatmap:** Built a complete $13 \times 13$ correlation matrix, highlighting strong relationships such as between `total_phenols` and `flavanoids` ($r = 0.86$).
3. **Feature Scaling:** Since K-Means calculates Euclidean distance, features with larger magnitudes (such as `proline` and `magnesium`) would dominate clustering. Standardized all 13 features using `StandardScaler` to have $\mu=0$ and $\sigma=1$.

---

## 🔍 Determining the Optimal Number of Clusters ($K$)

Tested cluster numbers across $K \in [2, 7]$:

1. **Inertia (Elbow Method):** Evaluates within-cluster sum of squares (WCSS).
2. **Silhouette Score:** Evaluates cluster cohesion versus separation distance.
3. **Dual-Axis Chart:** Plotted both metrics simultaneously on twin y-axes. The elbow curve showed a distinct inflection point at **$K = 3$**, accompanied by a robust silhouette score.

$$\text{Optimal } K = 3$$

---

## 🤖 Clustering Algorithms Comparison

With $K=3$, two partitioning strategies were trained and compared:

### 1. Standard K-Means
- Iterative centroids relocation (`n_init=10`, `random_state=42`).
- **Cluster Breakdown:**
  - Cluster 0: **65** wines
  - Cluster 1: **51** wines
  - Cluster 2: **62** wines

### 2. Bisecting K-Means
- Hierarchical top-down divisive approach recursively splitting clusters.
- **Cluster Breakdown:**
  - Cluster 0: **52** wines
  - Cluster 1: **61** wines
  - Cluster 2: **65** wines

---

## 📉 PCA Dimensionality Reduction & Visualization

To interpret and validate the 13-dimensional clusters visually:
- Applied **PCA (`n_components=2`)** to compress the 13 scaled features into two orthogonal axes (`PCA1` and `PCA2`).
- Plotted side-by-side scatter plots for **Standard K-Means** vs. **Bisecting K-Means** in 2D space.
- Both models successfully separate the data into three clean, well-isolated clusters that mirror the natural cultivars in the dataset.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Scikit-learn** (`KMeans`, `BisectingKMeans`, `PCA`, `StandardScaler`, `silhouette_score`)
- **Pandas & NumPy** (Data processing and array indexing)
- **Matplotlib & Seaborn** (Elbow curves, heatmaps, PCA scatter plots)

---

## 📁 Project Structure

```text
k_means&PCA_wine/
│
├── k_means&PCA_wine.ipynb   # Complete unsupervised clustering notebook
└── README.md                # Project documentation
```

---

## 🚀 How to Run

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook k_means&PCA_wine.ipynb
   ```
2. Run all cells sequentially. The dataset loads directly via `sklearn.datasets`.
