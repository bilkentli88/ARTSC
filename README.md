# ARTSC: Autoregressive Time Series Classification

## Overview
ARTSC is an R-based framework for time series classification that leverages autoregressive (AR) modeling to extract features and utilizes a Random Forest classifier for classification tasks. This method is designed to provide domain-independent solutions for time series data across diverse datasets.

---

## Features

- **Autoregressive Modeling**: Extracts AR coefficients, initial values, and fitting errors.
- **Dynamic Parameter Tuning**: Determines optimal lag and differencing values based on dataset complexity.
- **Random Forest Classification**: Employs a robust classifier to handle extracted features.
- **Dataset Complexity Scoring**: Adjusts parameters dynamically for computational efficiency.
- **Optional Feature Importance Module**: Helps analyze which features contribute the most to classification performance.

---

## Modules

### 1. `prepareDatasets.R`
Prepares datasets for the workflow by loading and organizing them into training and testing sets.

### 2. `fitARModel.R`
Fits an autoregressive model to the time series data and extracts features such as AR coefficients, initial values, and fitting errors.

### 3. `computeAccuracies.R`
Calculates classification accuracies based on extracted features using cross-validation with a Random Forest classifier.

### 4. `generateFeatures.R`
Generates a complete feature dataset using the optimal AR parameters (lag and differencing values).

### 5. `findBestParameters.R`
Identifies the optimal AR parameters (lag and differencing) for each dataset using grid search combined with cross-validation.

### 6. `computeImportance.R` (Optional)
Analyzes feature importance for curious coders, highlighting the most influential features for classification. Note: This module is not actively used in the core workflow.

---

## Workflow in RStudio

1. Load the modules:
   ```R
   source("Helpers/prepareDatasets.R")
   source("Helpers/fitARModel.R")
   source("Helpers/computeAccuracies.R")
   source("Helpers/generateFeatures.R")
   source("Helpers/findBestParameters.R")
   source("Helpers/computeImportance.R")  # Optional
   ```

2. Prepare the dataset:
   ```R
   prepareDatasets("data/GunPoint")
   ```

3. Find the best parameters:
   ```R
   best_params <- findBestParameters("GunPoint", p_max = 25, d_max = 3, verbose = TRUE)
   ```

4. Generate features:
   ```R
   generateFeatures(best_params)
   ```

5. Compute accuracies:
   ```R
   computeAccuracies(best_params)
   ```

6. (Optional) Analyze feature importance:
   ```R
   computeImportance("GunPoint")
   ```

---

## Requirements
- **RStudio**: Preferred environment for running the framework.
- **R Packages**: Ensure the following libraries are installed:
  - `randomForest`
  - `forecast`
  - `tseries`
  - `caret`
---

## Example Datasets
The framework has been tested on benchmark datasets from the UCR & UEA archive, including:

- **GunPoint** (Simple dataset)
- **Herring** (Moderate dataset)
- **Haptics** (Complex dataset)



## License
This project is licensed under the MIT License. See the LICENSE file for details.
