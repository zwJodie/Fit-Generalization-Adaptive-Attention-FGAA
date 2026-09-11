# Code for “Vulnerability-aware forecasting reduces excess yield forecast errors in low-yield regions”

## Overview

This directory contains the Python and R notebooks used for the analyses reported in the accompanying article. The workflow evaluates winter wheat yield prediction across yield strata and implements a vulnerability-aware forecasting framework that combines environmental and socioeconomic predictors with the Fit–Generalization Adaptive Attention (FGAA) algorithm. It covers feature construction, baseline training and FGAA-weighted fine-tuning for six deep learning models, Integrated Gradients (IG) attribution, temporal and regional robustness analyses, and preparation of the data used in the figures.

The workflow has three stages:

1. Python model training, evaluation and attribution (`code1.ipynb`–`code6.ipynb`)
2. Python aggregation of model outputs into figure-ready datasets (`book0.ipynb`)
3. R correlation, Mantel-test and network analyses (`book1.ipynb`–`book9.ipynb`)

## Expected project layout

Run the notebooks with `code/` as the working directory. The relative paths used in the notebooks assume the following layout:

```text
project/
├── data/
│   ├── README.md
│   ├── yield.csv
│   ├── social.csv
│   └── natural.csv
├── code/
│   ├── README.md
│   ├── function.py
│   ├── code0.ipynb ... code6.ipynb
│   └── book0.ipynb ... book9.ipynb
└── results/
    ├── work/
    │   └── 1/ ... 6/
    └── book/
        ├── 1/ ... 6/
        └── s1/ ... s6/
```

The code archive does not include the input datasets. Place `yield.csv`, `social.csv` and `natural.csv` in `../data/`. The notebooks create the required output directories under `../results/`.

## Software requirements

### Python

Use a recent Python 3 environment with the following packages:

```text
numpy
pandas
torch
scikit-learn
statsmodels
tqdm
captum
jupyter
```

For example:

```bash
python -m pip install numpy pandas torch scikit-learn statsmodels tqdm captum jupyter
```

PyTorch uses CUDA automatically when a compatible GPU is available. CPU execution is supported but is substantially slower.

### R

The R notebooks require an R kernel for Jupyter and the following packages:

```text
linkET
tidyverse
ggnewscale
RColorBrewer
viridis
scales
```

Install the CRAN packages in the usual way for your R environment. If `linkET` is unavailable for the installed R version, install it from its current source repository.

## Input data and fixed dimensions

The preprocessing workflow begins with 654 counties and 22 annual yield observations per county. For each county–year observation, eight environmental features are retained for each of nine growing-season months, giving 72 environmental inputs. Eight socioeconomic features are derived from nine county-level indicators.

### `yield.csv`

The first column contains county names. The remaining 22 columns contain annual winter wheat yield observations for 2001–2022. County names are not used as model inputs; counties receive zero-based internal indices according to their row order.

### `social.csv`

After the county-name column is removed, the nine indicators are reshaped into county × year × indicator order. The code derives the following eight socioeconomic features:

- Population Density (PopD)
- Health-care Resource (HeaR)
- Pupil Proportion (PuPr)
- Teenager Proportion (TePr)
- Fiscal Revenue (FiRv)
- Fiscal Expenditure (FiEx)
- Residential Savings (ReSv)
- Institutional Loans (InLn)

### `natural.csv`

The monthly environmental records are used to construct eight features:

- Radiation (Radn)
- Air Temperature (AirT)
- Soil Temperature (SoTe)
- Air Humidity (AirH)
- Soil Moisture (SoMo)
- Precipitation (Prec)
- Surface Pressure (SuPr)
- Wind Speed (WiSp)

Soil temperature and volumetric soil water are averaged across three layers, and wind speed is calculated from the 10-m zonal and meridional wind components. For each yield year, the notebooks retain the nine months from October of the preceding year through June of the harvest year.

## Shared implementation: `function.py`

`function.py` provides the data-processing, model-training, evaluation and attribution functions shared by the Python notebooks:

- `natural_feature_builder()` constructs the eight environmental features.
- `social_feature_builder()` derives the eight socioeconomic features.
- `filter_()` applies the common complete-case rule and retains counties with at least 10 complete annual observations.
- `flatten()` converts county–year arrays into sample-level matrices.
- `resolve()` decomposes yield observations (YO) into a LOWESS trend and discrepancy yield (DY).
- `scaler()` estimates standardization parameters from the training data and applies them to the validation and test data.
- `Model` implements a feedforward deep neural network (DNN), convolutional neural network (CNN), recurrent neural network (RNN), long short-term memory network (LSTM), gated recurrent unit (GRU) and attention-based neural network (Attn).
- `predictor()` performs baseline model training, sample-level FGAA score calculation, FGAA-weighted fine-tuning, evaluation and optional IG attribution.

For FGAA, the post-ReLU activation vector from the last hidden layer of the trained baseline model represents each training sample. Scaled dot-product similarities between representations are averaged across training samples to obtain a sample-level FGAA score. During fine-tuning, the precomputed scores are normalized within each batch and used as sample weights. Because the representations are post-ReLU, the resulting scores are nonnegative; the implementation requires their sum within each batch to be positive.

IG attribution uses an all-zero baseline in standardized feature space.

## Notebook guide

### Python experiment notebooks

- `code0.ipynb` generates `function.py`. Because the supplied module is ready to import, run this notebook only when regenerating the shared implementation; keep the notebook and module synchronized.
- `code1.ipynb` runs the full-period prediction experiment for six models and two feature settings. Outputs are written to `../results/work/1/`.
- `code2.ipynb` performs full-period IG analyses for yield observations (YO) and discrepancy yield (DY). It reuses the splits saved by `code1.ipynb` and writes outputs to `../results/work/2/`.
- `code3.ipynb` runs prediction experiments for eight provincial subsets. Outputs and split indices are written to `../results/work/3/`.
- `code4.ipynb` performs the provincial IG analyses. It reuses the splits from `code3.ipynb` and writes outputs to `../results/work/4/`.
- `code5.ipynb` runs prediction experiments for 2001–2011 and 2012–2022. Outputs and split indices are written to `../results/work/5/`.
- `code6.ipynb` performs the temporal IG analyses. It reuses the splits from `code5.ipynb` and writes outputs to `../results/work/6/`.

In saved model outputs, `baseline` refers to the 72 environmental inputs, whereas `addition` refers to the eight socioeconomic inputs followed by the 72 environmental inputs. Where all three training stages are retained, `init`, `pret` and `fine` denote the initial one-epoch model, the fully trained baseline model and the FGAA-enhanced model, respectively.

Reported metrics are the coefficient of determination (R²), mean squared error (MSE) and mean absolute error (MAE). R² is dimensionless; MSE and MAE are calculated on the standardized yield scale used for model fitting.

### Figure-data and R notebooks

- `book0.ipynb` combines outputs from `../results/work/` and creates the datasets under `../results/book/` used for the main and supplementary figures.
- `book1.ipynb` performs the full-study correlation, Mantel-test and network analyses.
- `book2.ipynb`–`book9.ipynb` perform the corresponding provincial analyses in this order: Hebei, Shanxi, Jiangsu, Anhui, Shandong, Henan, Hubei and Shaanxi.

## Recommended execution order

```text
code1.ipynb → code2.ipynb
code3.ipynb → code4.ipynb
code5.ipynb → code6.ipynb
                  ↓
              book0.ipynb
                  ↓
        book1.ipynb ... book9.ipynb
```

`code1.ipynb`, `code3.ipynb` and `code5.ipynb` can be run independently once the input files are available. Run each attribution notebook after its corresponding prediction notebook because the paired notebooks use the same saved split indices.

## Reproducibility notes

- The reported deep learning experiments used `max_epoch=1000` and `patience=20`; the Mantel tests used 999 permutations.
- The prediction notebooks generate random train/test and cross-validation splits without a fixed random seed. Exact reruns therefore depend on the saved split files: `../results/work/1/index.csv`, the province-specific index files under `../results/work/3/` and the period-specific index files under `../results/work/5/`.
- The attribution notebooks reuse these indices so that prediction and attribution results refer to the same held-out samples.
- Stochastic parameter initialization and GPU operations may produce small numerical differences between runs.
- The provincial analyses assume that the 654 counties are arranged consecutively as Hebei (117), Shanxi (76), Jiangsu (64), Anhui (57), Shandong (89), Henan (109), Hubei (67) and Shaanxi (75). If this order changes, update the province boundaries in `code3.ipynb` and `code4.ipynb`.
- Repeated cross-validation across six architectures requires substantial computation time and storage; GPU acceleration is strongly recommended.