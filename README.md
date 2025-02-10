# Estimation of galaxy physical parameters from Euclid photometry with CatBoost chained regression

## Overview
This repository contains the implementation of a machine-learning pipeline developed within the Euclid Collaboration to estimate galaxy physical properties using **CatBoost chained regressors and some other tricks**. The methodology is described in the paper:

> **Euclid Preparation: Estimating Galaxy Physical Properties Using CatBoost Chained Regressors with Attention**  
> *Euclid Collaboration: A. Humphrey et al. (2025), Astronomy & Astrophysics, in review.*

The pipeline is shared as a sanitised version of our rapidly-prototyped research code, which was designed and tested as part of a `data challenge' within the Euclid Collaboration. As set up here, it predicts galaxy redshift, stellar mass, star formation rate (SFR), specific SFR,  colour excess (E(B−V)), and stellar age using Euclid and ancillary photometric data. While our paper focuses on predicting these properties, the code can also be used to predict other galaxy properties, or indeed, the properties of other types of astronomical sources after minor modification. 

## Features
- **Gradient-boosted regression trees (CatBoost)** for high-performance property estimation, with GPU support.  
- **Chained regression approach**, where interdependent properties are predicted sequentially.  
- **Re-weighting attention mechanism** to optimize training data empirically.  
- **Uncertainty estimation** using computationally-efficient ML-predicted confidence intervals.  
- **Support for missing photometry**, making the model robust to real-world survey conditions.

## Code Structure
- `chained_regressors_pp_refactor_feb2025.py` – Main ML pipeline implementation.
- `params.yaml` - Parameter file needed to configure the pipeline.

## Example Data
[COSMOS2015_with_pdz_all_ebv.pq](https://drive.google.com/drive/folders/1if3M_UBgmO17ZDPS-U0Qt8njq8SvVhF5?usp=drive_link
)

## Data Format Requirements
- Input data needs to be in parquet format.
- At least 2 training features required (float).
- At least 2 target labels required (float).

## Installation & Dependencies
### Requirements
- Python 3.6.8+
- CatBoost
- NumPy
- Pandas
- Matplotlib
- Dask
- pyarrow
- Scikit-Learn

### Setup
Clone the repository and install dependencies:
```sh
git clone https://github.com/humphrey-and-the-machine/Euclid-chained-regression.git
cd Euclid-chained-regression
conda env create -f requirements.yaml 
```

## Usage
### Running the pipeline
```sh
python chained_regressors_pipeline.py params.yaml  
```




