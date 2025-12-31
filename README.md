# Dynamic Temperature Prediction of BEV Propulsion Motors

This repository contains the source code and experimental framework for my thesis: *"Dynamic Temperature Prediction Based on Physics-Informed Neural Networks."*

---

## 📌 Overview

Thermal management in Battery Electric Vehicle (BEV) propulsion motors is critical for performance and longevity. This study investigates the effectiveness of **Physics-Informed Neural Networks (PINNs)** compared to traditional **Lumped Parameter Thermal Networks (LPTNs)** in predicting temperatures under realistic production-like drive cycles.

### Research Focus

- **Sparse Data Performance**: Leveraging PINNs to predict temperatures with limited sensor measurements.  
- **Feature Integration**: Identifying and integrating system-level features for dynamic thermal behavior.  
- **Comparative Analysis**: Accuracy benchmarking between PINNs and LPTNs.  

## 📊 Dataset

The project utilizes the **Electric Motor Temperature Dataset** from Kaggle.

- **Key Targets**: Stator Winding, Stator Yoke, Stator Tooth, and Permanent Magnet temperatures.  
- **Conditions**: Realistic and production-like drive cycles of a permanent magnet synchronous motor (PMSM).  

## 🚀 Getting Started

## Development Environment

- **Operating System:** Ubuntu 22.04 
- **Python Version:** 3.10.12  
- **GPU (Recommended):** NVIDIA CUDA-enabled GPU for faster training of PINN and FNN models

## Installation

1. Clone the repository:

```bash
git clone https://github.com/cbohwalli/Physics-informed-Neural-Network-B.Sc-Thesis.git
cd Physics-informed-Neural-Network-B.Sc-Thesis
```

2. Create and Activate Your Environment

```bash
pyenv install 3.10.12
pyenv virtualenv 3.10.12 thesis-env
pyenv activate thesis-env
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🛠 Usage Pipeline

To reproduce the experiments, follow this sequence:

1. **Data Preparation:** Adds new features.

```bash
python src/preprocessing/prepare_dataset.py
```

2. **Exploratory Analysis:** Run correlation studies to understand feature importance.

```bash
python analysis/correlation_all.py
```

```bash
python analysis/correlation_target.py
```

3. **Running Experiments:** Execute the benchmark scripts located in the `experiments/` folder.

```bash
python experiments/pinn_benchmark.py
python experiments/lptn_benchmark.py
```

## 📂 Project Structure

```
.
├── analysis/           # Scripts for correlation and data studies
├── data/
│   ├── raw/            # Original measures_v2.csv
│   └── processed/      # Processed dataset used for analysis and experiments
├── experiments/        # Main benchmark execution scripts (PINN, LPTN, FNN)
├── results/        
├── src/
│   ├── models/         # Architecture definitions (PINN, LPTN, FNN)
│   └── preprocessing/  # Data pipeline and feature engineering
└── requirements.txt    # Project dependencies
```