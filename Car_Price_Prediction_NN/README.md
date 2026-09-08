# Car Price Prediction using Neural Networks

## 📌 Project Overview

This project is a regression project built as a hands-on **Mini Project** to practice writing a complete Deep Learning pipeline from scratch (no tutorials followed step-by-step).

The main goal is to apply a **Feed-Forward Neural Network** (Keras/TensorFlow) to predict car prices based on a set of numerical and categorical vehicle features.

The project also demonstrates important concepts, including:

- Nominal vs. Ordinal categorical encoding
- Train / Dev / Test splitting
- Feature Scaling without data leakage
- Neural Network architecture design
- Dropout & Regularization
- EarlyStopping
- Data-driven diagnosis of model performance

---

## 📊 Dataset

The dataset contains car listings with various numerical and categorical attributes used to predict the target variable `Price`.

### Features

- `Brand`
- `Year`
- `Engine Size`
- `Fuel Type`
- `Transmission`
- `Mileage`
- `Condition`
- `Model`

### Target

- `Price`

The original dataset contains **2500 samples and 10 columns** (including `Car ID`, later dropped).

There were no missing values and no duplicate rows detected.

---

## 🔍 Exploratory Data Analysis

Several EDA steps were performed to understand the dataset:

- Dataset inspection
- Missing values check
- Duplicate detection
- Categorical features value counts (`Brand`, `Fuel Type`, `Transmission`, `Condition`, `Model`)
- Target (`Price`) distribution analysis — histogram + boxplot
- Correlation analysis
- Correlation heatmap (top 10 features vs. `Price`)

The `Price` distribution was found to be approximately **uniform**, with no skew and no outliers.

---

## ⚙️ Data Preprocessing

The `Car ID` column was dropped as it is a non-informative identifier.

Categorical features were encoded based on their type:

- **Nominal** (`Brand`, `Fuel Type`, `Transmission`, `Model`) → **One-Hot Encoding**
- **Ordinal** (`Condition`: `Used` < `Like New` < `New`) → **Ordinal Encoding**

The data was divided into training, dev, and testing sets using a **70/15/15 split**.

Numerical features were standardized using `StandardScaler`.

The scaler was fitted only on the training data and then used to transform the dev and test data to avoid data leakage.

---

## 🤖 Model

The main model used in this project is:

**Feed-Forward Neural Network (Keras Sequential)**

```
Dense(128, relu) → Dropout(0.1)
Dense(64, relu)  → Dropout(0.1)
Dense(32, relu)  → Dropout(0.1)
Dense(1, linear)
```

- **Optimizer**: Adam (learning_rate = 0.001)
- **Loss**: Mean Squared Error (MSE)
- **Metric**: Mean Absolute Error (MAE)
- **Callback**: EarlyStopping (patience = 20, restore_best_weights = True)

---

## 📈 Model Evaluation

The model was evaluated using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

The final model achieved approximately:

- **Test MAE: ~$24,210**
- **Average Price: ~$52,912**
- **Error Rate: ~45.8%**

The training and validation loss curves were closely aligned throughout training, showing no clear evidence of overfitting.

---

## 🧠 Performance Analysis

Despite a technically correct pipeline (proper leakage-free scaling, appropriate encoding, regularization, and early stopping), the model's error rate (~46%) is far above what is acceptable for a production regression model.

Root cause analysis showed:

- **Near-zero correlation** between all features and `Price`
- **Uniform distribution** of `Price` across its full range, with no skew and no outliers — a pattern more consistent with randomly generated data than real-world car pricing

**Conclusion**: A model can only learn patterns that exist in the data. No amount of architecture tuning or hyperparameter optimization can compensate for a lack of real signal between features and target. This dataset's `Price` column appears to be synthetically/randomly generated, independent of the other features.

---

## 💡 Lesson Learned

Always check feature-target correlation **before** investing time in model building. If there is no signal in the data, the priority shifts to sourcing better data — not tuning the model further.

---

## 🧰 Tech Stack

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras
