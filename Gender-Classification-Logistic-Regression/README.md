# Gender Classification using Logistic Regression

## 📌 Project Overview

This project is a simple binary classification project built as part of the **NTI Machine Learning Training**.

The main goal is to apply **Logistic Regression** to predict gender based on a set of facial measurements and binary facial features.

The project also demonstrates important concepts covered in the lecture, including:

- Logistic Regression
- Sigmoid Function
- Probability Prediction
- Cross Entropy / Log Loss
- Overfitting
- Regularization

---

## 📊 Dataset

The dataset contains facial measurements and characteristics that are used to predict the target variable `gender`.

### Features

- `long_hair`
- `forehead_width_cm`
- `forehead_height_cm`
- `nose_wide`
- `nose_long`
- `lips_thin`
- `distance_nose_to_lip_long`

### Target

- `gender`

The original dataset contains **5001 samples and 8 columns**.

There were no missing values, but duplicate rows were detected and removed before training the model.

---

## 🔍 Exploratory Data Analysis

Several EDA steps were performed to understand the dataset:

- Dataset inspection
- Missing values check
- Duplicate detection
- Target distribution analysis
- Boxplots for numerical features
- Correlation analysis
- Correlation heatmap

After removing duplicate rows, the dataset contained **3233 unique samples**.

---

## ⚙️ Data Preprocessing

The data was divided into training and testing sets using an **80/20 split**.

Numerical features were standardized using `StandardScaler`.

The scaler was fitted only on the training data and then used to transform the test data to avoid data leakage.

---

## 🤖 Model

The main model used in this project is:

**Logistic Regression**

The model predicts the probability of belonging to a gender class and converts the probability into a class prediction.

---

## 📈 Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Log Loss (Cross Entropy)

The final model achieved approximately:

- **Test Accuracy: 95.98%**
- **Test Log Loss: 0.102**

The classification performance was balanced across both classes.

---

## 🧠 Overfitting Analysis

Training and testing performance were compared to check for overfitting.

The difference between training and testing accuracy was small, so there was **no clear evidence of significant overfitting** in the final Logistic Regression model.

---

## 🛡️ Regularization

Regularization was studied by testing different values of the Logistic Regression parameter `C`:

```text
0.01, 0.1, 1, 10, 100
