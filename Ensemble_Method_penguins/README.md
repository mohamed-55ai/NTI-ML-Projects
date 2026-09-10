# 🐧 Palmer Penguins Body Mass Prediction using Ensemble Methods

## 📌 Project Overview

This project explores and compares classical regression and modern **Ensemble Learning** techniques on the **Palmer Penguins** dataset. 

The objective is to accurately predict penguin body mass (`body_mass_g`) using biometric measurements and demographic data, benchmarking three distinct machine learning paradigms:
1. **Linear Regression** (Standard Parametric Baseline)
2. **Bagging Regressor** (Bootstrap Aggregation with Decision Trees)
3. **XGBoost Regressor** (Gradient Tree Boosting)

---

## 📊 Dataset

The dataset is loaded directly using Seaborn (`sns.load_dataset("penguins")`).

- **Total Samples:** 344 observations
- **Target Variable:** `body_mass_g` (Penguin body mass in grams)

### Features

| Feature | Type | Description |
| :--- | :--- | :--- |
| `species` | Categorical | Penguin species (`Adelie`, `Chinstrap`, `Gentoo`) |
| `island` | Categorical | Island in Palmer Archipelago (`Biscoe`, `Dream`, `Torgersen`) |
| `bill_length_mm` | Numerical | Length of the culmen / bill (mm) |
| `bill_depth_mm` | Numerical | Depth of the culmen / bill (mm) |
| `flipper_length_mm` | Numerical | Flipper length (mm) |
| `sex` | Categorical | Penguin sex (`Male`, `Female`) |

---

## ⚙️ Data Preprocessing & EDA

1. **Handling Missing Data:** Checked for null values (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`: 2 nulls each; `sex`: 11 nulls). Dropped missing rows, resulting in **333 clean observations**.
2. **Categorical Encoding:** Applied One-Hot Encoding with `drop_first=True` on `species`, `island`, and `sex` to eliminate dummy variable trap.
3. **Correlation Analysis:** Generated a correlation heatmap showing strong positive correlation between `flipper_length_mm` and `body_mass_g`.
4. **Train / Test Split:** Divided data into **80% training** (266 samples) and **20% testing** (67 samples) with `random_state=42`.
5. **Feature Scaling:** Standardized input features using `StandardScaler` to prevent feature scale bias.

---

## 🤖 Models & Configurations

### 1. Linear Regression
- Classical ordinary least squares regression fitted on standardized features.

### 2. Bagging Regressor
- **Base Estimator:** `DecisionTreeRegressor(random_state=42)`
- **Number of Estimators (`n_estimators`):** 300
- **Max Samples:** 80% (`max_samples=0.8`)
- **Parallelization:** `n_jobs=-1`

### 3. XGBoost Regressor
- **Objective:** `reg:squarederror`
- **Number of Estimators (`n_estimators`):** 300
- **Learning Rate:** 0.05
- **Parallelization:** `n_jobs=-1`

---

## 📈 Evaluation & Results Comparison

Models were evaluated on the unseen test set using **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and the coefficient of determination (**$R^2$ Score**):

| Model | MAE (g) | RMSE (g) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** ⭐ | **196.209** | **255.749** | **0.896** |
| **Bagging Regressor** | 232.124 | 283.795 | 0.872 |
| **XGBoost (Boosting)** | 259.339 | 316.160 | 0.841 |

### 💡 Key Takeaways & Insights

- **Linear Regression performed best** with the lowest error ($MAE \approx 196\text{g}$) and highest $R^2$ ($0.896$).
- The relationship between morphological variables (especially flipper length) and penguin body mass is strongly linear, making linear models highly effective and less prone to variance on smaller tabular sample sizes ($N=333$).
- Both **Bagging** and **XGBoost** achieved robust performance ($R^2 > 0.84$), demonstrating solid predictive power across different ensemble strategies.

### Visualizations Included in Notebook

- **Comparison Bar Chart:** Side-by-side visualization of MAE, RMSE, and $R^2$ across all three models.
- **Actual vs. Predicted Scatter Plots:** Three individual subplots comparing predictions against actual test targets with a $45^\circ$ reference line.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Scikit-learn** (Linear Regression, Bagging, StandardScaler, Metrics)
- **XGBoost** (Gradient Boosting)
- **Pandas & NumPy** (Data wrangling and matrix operations)
- **Seaborn & Matplotlib** (Statistical visualizations and plots)

---

## 📁 Project Structure

```text
Ensemble_Method_penguins/
│
├── Ensemble_Method_penguins.ipynb   # Complete analysis and modeling notebook
└── README.md                         # Project documentation
```

---

## 🚀 How to Run

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Ensemble_Method_penguins.ipynb
   ```
2. Run all cells in order. The dataset loads directly via Seaborn without external CSV requirements.
