# Fit–Generalization Adaptive Attention (FGAA)-based Crop Yield Prediction with Environmental and Socioeconomic Features

## Overview

This repository contains the code for **crop yield prediction**, integrating **environmental and socioeconomic factors**. The core algorithm used is **FGAA**, which adaptively up-weights underfitted extreme samples during training.

The workflow integrates:

- environmental and socioeconomic feature construction
- deep learning–based yield prediction
- FGAA algorithm designed for extreme yield prediction
- feature attribution analysis
- statistical association analysis
- visualization and figure generation

The repository combines **Python-based deep learning experiments** and **R-based statistical analysis and visualization**.

The complete workflow consists of three stages:

1. **Model training and prediction** (Python)  
2. **Result aggregation and dataset construction** (Python)  
3. **Statistical analysis and visualization** (R)  

---

# Repository Structure

```
.
├── data/
│   ├── yield.csv
│   ├── social.csv
│   └── natural.csv
│
├── function.py
│
├── Code0.ipynb
├── Code1.ipynb
├── Code2.ipynb
├── Code3.ipynb
├── Code4.ipynb
├── Code5.ipynb
├── Code6.ipynb
│
├── Book0.ipynb
├── Book1.ipynb
├── Book2.ipynb
├── Book3.ipynb
├── Book4.ipynb
├── Book5.ipynb
├── Book6.ipynb
├── Book7.ipynb
├── Book8.ipynb
├── Book9.ipynb
│
├── work/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
│   └── 6/
│
└── book/
    ├── 1/
    ├── 2/
    ├── 3/
    ├── 4/
    ├── 5/
    ├── 6/
    ├── s1/
    ├── s2/
    ├── s3/
    ├── s4/
    ├── s5/
    └── s6/
```

---

# Environment

## Python

Recommended version:

```
Python >= 3.7
PyTorch >= 1.12
```

Main dependencies:

```
numpy
pandas
torch
scikit-learn
statsmodels
tqdm
captum
d2l
```

Install dependencies:

```bash
pip install numpy pandas torch scikit-learn statsmodels tqdm captum d2l
```

GPU acceleration is recommended but not required.

---

## R

Required packages:

```
linkET
tidyverse
ggnewscale
RColorBrewer
viridis
```

Install packages:

```r
install.packages(c("linkET","tidyverse","ggnewscale","RColorBrewer","viridis"))
```

---

# Data

The `data/` directory contains three datasets.

## yield.csv

County-level yield observations.

Structure:

```
county × year
```

---

## social.csv

Socioeconomic indicators, including:

- **PopS** — population size
- **HeaR** — health resources
- **PuPr** — pupil proportion
- **TePr** — teenager proportion
- **FiRv** — fiscal revenue
- **FiEx** — fiscal expenditure
- **ReSv** — residential savings
- **InLn** — institutional loans

---

## natural.csv

Environmental variables including:

- **Radn** — solar radiation
- **AirT** — air temperature
- **SoTe** — soil temperature
- **AirH** — air humidity
- **SoMo** — soil moisture
- **Prec** — precipitation
- **SuPr** — surface pressure
- **WiSp** — wind speed

---

# Core Python Module

## function.py

Contains reusable functions used across experiments.

### Feature construction

```
natural_feature_builder()
social_feature_builder()
```

### Data preprocessing

```
filter_()
flatten()
resolve()
scaler()
```

These functions perform:

- missing-value filtering
- dataset reshaping
- trend–residual decomposition
- feature standardization

---

## Deep Learning Models

The following neural architectures are implemented:

- DNN
- CNN
- RNN
- LSTM
- GRU
- Attention

---

## Training Pipeline

```
predictor()
```

Implements a **two-stage training framework**:

1. Initial model training
2. FGAA fine-tuning

---

## Model Interpretation

```
explainer()
```

Supported attribution methods:

- Integrated Gradients
- GradientShap

---

# Model Training Experiments

The **Code notebooks** implement the deep learning experiments.

---

## Code0.ipynb

Exports shared functions to:

```
function.py
```

---

## Code1.ipynb

Main yield prediction experiment.

Tasks include:

- feature construction  
- dataset preprocessing  
- model training  
- performance evaluation  

Two feature settings are tested:

- **baseline**: environmental features only  
- **addition**: environmental + socioeconomic features  

Outputs are stored in:

```
work/1/
```

---

## Code2.ipynb

Feature attribution experiment.

Computes feature importance for:

- observed yield
- residual yield

Using:

- Integrated Gradients  
- GradientShap  

Outputs:

```
work/2/
```

---

## Code3.ipynb

Regional prediction experiment.

The dataset is divided into eight provinces:

- Hebei
- Shanxi
- Jiangsu
- Anhui
- Shandong
- Henan
- Hubei
- Shaanxi

Outputs:

```
work/3/
```

---

## Code4.ipynb

Regional feature attribution analysis.

Outputs:

```
work/4/
```

---

## Code5.ipynb

Intertemporal prediction experiment.

Two time periods are analyzed:

- early period (2001-2011)
- later period (2012-2022)

Outputs:

```
work/5/
```

---

## Code6.ipynb

Intertemporal feature attribution analysis.

Outputs:

```
work/6/
```

---

# Figure Generation

The **Book notebooks** convert model outputs into datasets used for figures and supplementary figures.

---

## Book0.ipynb

Aggregates model outputs from `work/` and generates figure datasets in:

```
book/
```

Outputs include:

- yield statistics  
- prediction errors  
- feature importance  
- province-level datasets  
- temporal datasets  

These datasets are used for all figures in the manuscript.

---

## Book1.ipynb

Computes **Mantel tests** and **correlation matrices** between yield variables and environmental/socioeconomic features.

Outputs:

```
book/2/
```

Also generates the **network heatmap visualization** combining:

- Pearson correlations  
- Mantel statistics  

---

## Book2–Book9.ipynb

Perform province-level Mantel tests and correlation analyses.

Each notebook corresponds to one province:

- Hebei
- Shanxi
- Jiangsu
- Anhui
- Shandong
- Henan
- Hubei
- Shaanxi

Outputs include:

- Mantel test results
- correlation matrices
- network heatmap plots

Stored in:

```
book/s2/
```

---

# Outputs

Two main result directories are produced.

---

## work/

Stores **raw experimental outputs from deep learning models**.

Typical files include:

```
train_inst_*.csv
test_inst_*.csv
train_accu_*.csv
test_accu_*.csv
*_ig_*.csv
*_gs_*.csv
```

These files contain:

- prediction errors
- model accuracy
- feature attribution scores
- cross-validation results

---

## book/

Stores **processed datasets used for figures**.

Examples:

```
yield_grouped.csv
error_grouped.csv
feature_book.csv
mantel_result.csv
correl_matrix.csv
```

Also includes **figure-ready data for supplementary materials**.

---

# Reproducibility

To reproduce all results:

## Step 1 — Run deep learning experiments

Execute notebooks sequentially:

```
Code0.ipynb
Code1.ipynb
Code2.ipynb
Code3.ipynb
Code4.ipynb
Code5.ipynb
Code6.ipynb
```

Outputs will be written to:

```
work/
```

---

## Step 2 — Generate figure datasets

Run:

```
Book0.ipynb
```

Outputs will be written to:

```
book/
```

---

## Step 3 — Statistical analysis and visualization

Run:

```
Book1.ipynb
Book2.ipynb
Book3.ipynb
...
Book9.ipynb
```

These scripts perform:

- Mantel tests  
- correlation analysis  
- network visualization  

---

# Notes

- Deep learning experiments may require **substantial runtime** due to repeated model training and cross-validation.
- **GPU acceleration is recommended** for faster training.
- Minor numerical variation may occur due to **stochastic optimization**.