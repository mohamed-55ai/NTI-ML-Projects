# ⏱️ Sequence Modeling & Time-Series Forecasting: SimpleRNN vs. LSTM vs. GRU

## 📌 Project Overview

This project provides an empirical benchmark comparing three fundamental **Recurrent Neural Network (RNN)** architectures for sequence learning and time-series forecasting:
1. **SimpleRNN** (Vanilla Recurrent Neural Network)
2. **LSTM** (Long Short-Term Memory)
3. **GRU** (Gated Recurrent Unit)

Implemented using **TensorFlow / Keras**, all three architectures are evaluated under identical training configurations on a synthetic sequential dataset to study parameter efficiency, convergence behavior, and predictive error.

---

## 📊 Problem Formulation & Data Generation

The project synthesizes a continuous time-series signal modeled as a noisy sine wave:

$$y(t) = \sin(t) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 0.01)$$

- **Time Horizon:** $t \in [0, 50]$ with 500 evenly spaced points (`np.linspace(0, 50, 500)`).
- **Sequence Generation:** Formulated using a sliding window of length **$10$** (`sequence_length = 10`):
  - Input: Past 10 consecutive time steps $\rightarrow$ Shape: `(N, 10, 1)`
  - Output: The subsequent 11th step $\rightarrow$ Shape: `(N,)`
- **Data Splitting:**
  - **Training set:** 392 sequences (80%)
  - **Testing / Validation set:** 98 sequences (20%)

---

## 🤖 Architectures & Model Design

To ensure rigorous benchmarking, each model employs an identical single-recurrent layer topology followed by a single output unit:

```text
Input (Batch, 10, 1)
  │
  ├── [SimpleRNN | LSTM | GRU] (32 units)
  │
  └── Dense(1, linear)
```

### Parameter Count Comparison

| Model | Gating Mechanism | Total Parameters | Memory Footprint |
| :--- | :--- | :---: | :---: |
| **SimpleRNN** | Single hidden state transformation | **1,121** | 4.38 KB |
| **GRU** | Reset & Update gates | **3,361** | 13.13 KB |
| **LSTM** | Forget, Input, Output & Cell state gates | **4,385** | 17.13 KB |

### Training Hyperparameters

- **Loss Function:** Mean Squared Error (`mse`)
- **Optimizer:** Adam
- **Epochs:** 20
- **Batch Size:** 16
- **Validation:** Evaluated against test sequences after each epoch

---

## 📈 Benchmark Results

### Final Validation Performance (MSE)

| Model | Validation MSE | Convergence & Characteristics |
| :--- | :---: | :--- |
| **SimpleRNN** ⭐ | **0.0161** | Lowest error and fastest training on short sequence windows ($L=10$). |
| **GRU** | **0.0163** | Near-identical precision with 23% fewer parameters than LSTM. |
| **LSTM** | **0.0174** | Strong stability; optimal for long-range sequence dependencies. |

### 💡 Engineering Insights

- For short-context window tasks ($L=10$ steps) without vanishing gradient challenges, **SimpleRNN** is highly effective and computationally lightweight.
- **GRU** offers an optimal middle-ground: capturing gated memory dynamics with significantly lower compute requirements than standard LSTM.
- A comparative validation loss curve across all 20 epochs illustrates the rapid learning trajectory of all three architectures.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow / Keras** (Recurrent network layers and training)
- **NumPy** (Sequence windowing and signal generation)
- **Matplotlib** (Signal and loss curve plotting)

---

## 📁 Project Structure

```text
RNN&LTSM&GRU/
│
├── RNN&LTSM&GRU_task.ipynb   # Complete time-series modeling notebook
└── README.md                 # Project documentation
```

---

## 🚀 How to Run

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook RNN&LTSM&GRU_task.ipynb
   ```
2. Run all cells sequentially. The dataset is generated in-memory deterministically (`random_state = 42`).
