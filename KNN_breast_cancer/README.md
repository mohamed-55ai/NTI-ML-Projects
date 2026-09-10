# 🎗️ Breast Cancer Detection & Classification with Threshold Optimization

## 📌 Project Overview

This project focuses on automated diagnosis of breast cancer from digitized cell nuclei measurements using **Supervised Machine Learning**. 

The goal is to accurately classify tumors as **Malignant** (0) or **Benign** (1). Beyond comparing standard classifiers (**KNN**, **SVM**, and **Logistic Regression**), the project emphasizes clinical decision-making through **Decision Threshold Optimization** to maximize diagnostic F1-Score.

---

## 📊 Dataset

The project uses the widely benchmarked **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset via `sklearn.datasets.load_breast_cancer`.

- **Total Samples:** 569 instances
- **Features:** 30 numeric, real-valued biological features describing characteristics of cell nuclei:
  - 10 Mean features (`mean radius`, `mean texture`, `mean perimeter`, `mean area`, `mean smoothness`, `mean compactness`, `mean concavity`, `mean concave points`, `mean symmetry`, `mean fractal dimension`)
  - 10 Error / Standard Error features (`radius error`, `texture error`, etc.)
  - 10 "Worst" / largest value features (`worst radius`, `worst texture`, etc.)
- **Target Variable:** `target`
  - `0`: **Malignant** (212 samples, ~37.3%)
  - `1`: **Benign** (357 samples, ~62.7%)

---

## ⚙️ Exploratory Data Analysis & Preprocessing

1. **Data Health:** Zero missing values across all 30 features.
2. **Class Balance Check:** Target distribution visualized using Seaborn countplot.
3. **Correlation Analysis:** Generated a correlation heatmap focusing on top features vs. target, showing strong inverse correlations between malignant tumors and cell dimensions (e.g., higher radius, perimeter, and area correlate with malignancy).
4. **Distribution Checks:** Comparative boxplots of `mean radius` and `mean texture` grouped by tumor malignancy.
5. **Standardization:** All 30 features standardized using `StandardScaler`.
6. **Train / Test Split:** **80% training** (455 samples) and **20% testing** (114 samples) with `random_state=42`.

---

## 🤖 Models & Hyperparameter Tuning

### 1. K-Nearest Neighbors (KNN - Tuned)
- Hyperparameter tuning via **GridSearchCV** with 5-fold cross-validation:
  - `n_neighbors`: `[3, 5, 7, 9, 11]`
  - `weights`: `['uniform', 'distance']`

### 2. Support Vector Machine (SVM)
- **Kernel:** Radial Basis Function (`rbf`)
- **Probability:** `True`
- **Random State:** 42

### 3. Logistic Regression
- **Max Iterations:** 1000
- **Random State:** 42

---

## 📈 Model Comparison Results

| Model | Train Accuracy | Test Accuracy | Status |
| :--- | :---: | :---: | :---: |
| **SVM (RBF)** | 98.68% | **97.37%** | Top Performer |
| **Logistic Regression** | 98.68% | **97.37%** | Top Performer |
| **KNN (Tuned)** | 100.00% | 94.74% | High Performance |

A side-by-side grouped bar chart illustrates train vs. test accuracy across all three models.

---

## 🎯 Threshold Optimization for Medical Decision-Making

In healthcare applications, standard 0.50 probability cutoff may not yield optimal balance between precision and recall:

- Evaluated classification thresholds ranging from **0.10 to 0.85** in steps of 0.05 on Logistic Regression predicted probabilities.
- **Default Threshold (0.50) F1-Score:** `0.9790`
- **Optimized Threshold (0.25) F1-Score:** `0.9861` ⭐

Visualized via a threshold optimization plot with the red dashed indicator marking the peak diagnostic threshold at `0.25`.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Scikit-learn** (Classifiers, GridSearchCV, StandardScaler, Metrics)
- **Pandas & NumPy** (Data processing)
- **Matplotlib & Seaborn** (Data visualization)

---

## 📁 Project Structure

```text
KNN_breast_cancer/
│
├── KNN_breast_cancer.ipynb   # Complete exploratory and modeling notebook
└── README.md                 # Project documentation
```

---

## 🚀 How to Run

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook KNN_breast_cancer.ipynb
   ```
2. Execute all notebook cells sequentially. The dataset loads directly from `scikit-learn`.
