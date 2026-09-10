# 🖼️ Fashion-MNIST Image Classification using Convolutional Neural Networks (CNN)

## 📌 Project Overview

This project implements an end-to-end Deep Learning pipeline for multi-class image classification using **Convolutional Neural Networks (CNN)** built with **TensorFlow / Keras**. 

> [!NOTE]
> Although the directory is titled `CNN_CIFAR10_Image_Classification`, the notebook implements and evaluates the CNN architecture on the **Fashion-MNIST** dataset.

The objective is to accurately categorize $28 \times 28$ grayscale images of fashion items into one of 10 distinct clothing and footwear classes.

---

## 📊 Dataset

The project uses the **Fashion-MNIST** dataset, loaded directly via `tensorflow.keras.datasets.fashion_mnist`.

- **Training samples:** 60,000 images
- **Testing samples:** 10,000 images
- **Image dimensions:** $28 \times 28$ pixels, grayscale
- **Number of classes:** 10

### Class Names

| Label | Class Name | Label | Class Name |
| :---: | :--- | :---: | :--- |
| **0** | T-shirt/top | **5** | Sandal |
| **1** | Trouser | **6** | Shirt |
| **2** | Pullover | **7** | Sneaker |
| **3** | Dress | **8** | Bag |
| **4** | Coat | **9** | Ankle boot |

---

## ⚙️ Data Preprocessing & EDA

1. **Channel Dimension Expansion:** Added channel axis to grayscale images using `np.expand_dims` $\rightarrow$ shape `(N, 28, 28, 1)`.
2. **Pixel Normalization:** Scaled pixel values from $[0, 255]$ to $[0.0, 1.0]$ by dividing by `255.0` (`float32`).
3. **Data Visualization:** Plotted a $2 \times 5$ grid displaying sample clothing images along with their respective ground-truth class labels.

---

## 🤖 Model Architecture

A sequential Convolutional Neural Network designed for spatial feature extraction followed by classification:

```text
Input (28, 28, 1)
  │
  ├── Conv2D (32 filters, 3x3 kernel, ReLU)        --> (26, 26, 32)
  ├── MaxPooling2D (2x2 pool size)                  --> (13, 13, 32)
  │
  ├── Conv2D (64 filters, 3x3 kernel, ReLU)        --> (11, 11, 64)
  ├── MaxPooling2D (2x2 pool size)                  --> (5, 5, 64)
  │
  ├── Flatten                                      --> (1600)
  ├── Dense (64 units, ReLU)                       --> (64)
  └── Dense (10 units, Softmax)                    --> (10)
```

### Parameter Summary

- **Total Parameters:** 121,930 (476.29 KB)
- **Trainable Parameters:** 121,930
- **Non-trainable Parameters:** 0

### Training Configuration

- **Optimizer:** Adam
- **Loss Function:** `sparse_categorical_crossentropy`
- **Metric:** `accuracy`
- **Epochs:** 5
- **Batch Size:** 32 (default, 1875 steps per epoch)

---

## 📈 Results & Performance

### Training Progression across 5 Epochs

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.4675 | 82.99% | 0.3711 | 86.64% |
| **2** | 0.3127 | 88.56% | 0.3204 | 88.72% |
| **3** | 0.2659 | 90.24% | 0.2908 | 89.03% |
| **4** | 0.2354 | 91.33% | 0.2686 | 90.18% |
| **5** | 0.2142 | 92.01% | 0.2671 | 90.26% |

### Final Evaluation on Unseen Test Set

- **Test Loss:** `0.2671`
- **Test Accuracy:** `90.26%`

Visual learning curves for both accuracy and loss across epochs demonstrate smooth convergence without severe overfitting.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow / Keras** (Deep Learning framework)
- **NumPy** (Numerical arrays and transformations)
- **Matplotlib** (Visualization and performance plots)

---

## 📁 Project Structure

```text
CNN_CIFAR10_Image_Classification/
│
├── CNN_CIFAR10_Image_Classification.ipynb   # Complete training & evaluation notebook
└── README.md                                # Project documentation
```

---

## 🚀 How to Run

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook CNN_CIFAR10_Image_Classification.ipynb
   ```
2. Execute all cells sequentially. The dataset will be downloaded automatically by Keras on the first run.
