# 🎃 Pumpkin Seeds Classification using Support Vector Machines (SVM)

## 📌 Project Overview

This project provides an automated classification system for distinguishing between two commercially important cultivars of pumpkin seeds—**Çerçevelik** and **Ürgüp Sivrisi**—using geometrical and morphological attributes extracted from digital images.

The pipeline applies **Support Vector Machines (SVM)** integrated into a Scikit-learn pipeline and optimized using **Stratified 5-Fold GridSearchCV** across multiple scalers, kernels, and regularization strengths.

---

## 📊 Dataset

The dataset is stored locally in `Pumpkin_Seeds_Dataset.csv` and contains morphological features for 2,500 pumpkin seeds.

- **Total Samples:** 2,500
- **Missing Values:** 0
- **Duplicate Rows:** 0
- **Target Distribution:**
  - `Çerçevelik`: 1,300 samples (52.0%) $\rightarrow$ Mapped to `0`
  - `Ürgüp Sivrisi`: 1,200 samples (48.0%) $\rightarrow$ Mapped to `1`

### Morphological Features (12 attributes)

| Feature | Description |
| :--- | :--- |
| `Area` | Number of pixels within the seed boundary |
| `Perimeter` | Circumference / boundary length of the seed |
| `Major_Axis_Length` | Length of the main ellipse axis through the seed |
| `Minor_Axis_Length` | Length of the minor ellipse axis |
| `Convex_Area` | Smallest convex polygon area bounding the seed |
| `Equiv_Diameter` | Diameter of a circle with equivalent area |
| `Eccentricity` | Non-circularity measure of the seed ellipse |
| `Solidity` | Ratio of seed area to convex polygon area |
| `Extent` | Ratio of seed area to bounding box area |
| `Roundness` | Measure of circular shape symmetry |
| `Aspect_Ration` | Ratio of major axis length to minor axis length |
| `Compactness` | Measure of the degree to which a shape is compact |

---

## ⚙️ Exploratory Data Analysis & Preprocessing

1. **Correlation Analysis:** Analyzed feature correlations with seed class:
   - **Strong Positive Correlation with Ürgüp Sivrisi (Class 1):** `Aspect_Ration` (+0.72), `Eccentricity` (+0.70), `Major_Axis_Length` (+0.56).
   - **Strong Negative Correlation with Ürgüp Sivrisi (Class 1):** `Compactness` (-0.73), `Roundness` (-0.67), `Minor_Axis_Length` (-0.40).
2. **Linear Separability Check:** Scatter plot of `Aspect_Ration` vs `Compactness` highlighted distinct cluster boundaries with partial non-linear overlap.
3. **Train / Test Split:** **80% training** (2,000 samples) and **20% testing** (500 samples) with `random_state=42`.

---

## 🤖 Modeling Pipeline & Hyperparameter Tuning

A full Scikit-learn `Pipeline` was established:
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])
```

### Grid Search Strategy
- **Cross-Validation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Evaluation Metric:** `accuracy`
- **Search Space:**
  - **Scalers:** `StandardScaler()`, `RobustScaler()`
  - **RBF Kernel:** `C: [1, 5, 10, 20, 50, 100]`, `gamma: ['scale', 'auto', 0.01, 0.05, 0.1, 0.2]`
  - **Linear Kernel:** `C: [0.1, 1, 10, 50]`

### Best Discovered Hyperparameters

- **Scaler:** `StandardScaler()`
- **Kernel:** `rbf`
- **C:** `100`
- **Gamma:** `0.05`
- **Best 5-Fold Cross-Validation Accuracy:** **89.60%**

---

## 📈 Test Evaluation & Results

The best estimator was evaluated on the unseen test set (500 samples):

- **Test Accuracy:** `86.80%`
- **ROC-AUC Score:** `0.9373` ⭐

### Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Çerçevelik (0)** | 0.85 | 0.89 | 0.87 | 251 |
| **Ürgüp Sivrisi (1)** | 0.89 | 0.84 | 0.86 | 249 |
| **Overall Accuracy** | | | **0.87** | 500 |
| **Macro Average** | 0.87 | 0.87 | 0.87 | 500 |
| **Weighted Average** | 0.87 | 0.87 | 0.87 | 500 |

### Confusion Matrix
A heatmap visualization depicts clear separation between cultivars with low misclassification rates across both seed varieties.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Scikit-learn** (Pipeline, SVC, GridSearchCV, StratifiedKFold, Scalers, Metrics)
- **Pandas & NumPy** (Data wrangling and feature manipulation)
- **Seaborn & Matplotlib** (Statistical heatmaps and distribution plots)

---

## 📁 Project Structure

```text
Pumpkin-Seeds-SVM/
│
├── Pumpkin_Seeds_Classification_SVM.ipynb   # Complete analysis and modeling notebook
├── Pumpkin_Seeds_Dataset.csv                # Dataset file (2500 samples)
└── README.md                                # Project documentation
```

---

## 🚀 How to Run

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Pumpkin_Seeds_Classification_SVM.ipynb
   ```
2. Execute the cells. The notebook will automatically read `Pumpkin_Seeds_Dataset.csv` from the same directory and execute the grid search pipeline.
