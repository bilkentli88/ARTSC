# ARTSC
# ARTSC: Autoregressive Time Series Classification

## Project Overview
ARTSC (Autoregressive Time Series Classification) is a machine learning framework for time series classification. The method leverages autoregressive (AR) modeling techniques and Random Forest (RF) classifiers to provide a domain-independent approach for analyzing time series data.

The framework is implemented in R and includes six modules, each designed to handle a specific aspect of the workflow, from dataset preparation to feature extraction and model evaluation. ARTSC is intended for researchers and practitioners who require an efficient and versatile method for time series classification.

## Modules
### Essential Modules
1. **generateFeatures.R**
   - Responsible for extracting features from the time series data based on AR modeling.
   - Outputs a feature dataset that includes AR coefficients, total fitting error, and raw values.

2. **findBestParameters.R**
   - Determines the optimal lag (`p`) and differencing (`d`) parameters for AR modeling through grid search and cross-validation.
   - Employs a dynamic strategy to handle non-stationary datasets using KPSS tests.

### Helper Modules
1. **prepareDatasets.R**
   - Prepares the input datasets by splitting them into training and testing sets.
   - Includes preprocessing steps such as normalization and basic dataset integrity checks.

2. **fitARModel.R**
   - Fits an autoregressive model to the input data using the specified `p` and `d` parameters.
   - Outputs AR coefficients and total fitting error.

3. **computeAccuracies.R**
   - Computes classification accuracies using the extracted features and Random Forest classifier.
   - Supports dynamic RF parameter adjustments based on dataset complexity.

4. **computeImportance.R** (Optional)
   - Analyzes the importance of features for curious developers.
   - Outputs feature importance plots to provide insights into the model.

## How to Use
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Set up your environment:
   - Ensure R and required packages (e.g., `randomForest`, `forecast`) are installed.

3. Run the modules in the following order for a complete workflow:
   1. `prepareDatasets.R`
   2. `findBestParameters.R`
   3. `fitARModel.R`
   4. `generateFeatures.R`
   5. `computeAccuracies.R`

   **Note:** The `computeImportance.R` module is optional and can be used for feature analysis.

4. Customize parameters in `findBestParameters.R` (e.g., dataset name, maximum lag value, verbosity).

## Example
Here is an example of how to run the modules in sequence:
```R
# Load required scripts
source("prepareDatasets.R")
source("findBestParameters.R")
source("fitARModel.R")
source("generateFeatures.R")
source("computeAccuracies.R")

# Run the workflow
prepareDatasets("data/Coffee")
best_params <- findBestParameters("Coffee", p_max = 25, d_max = 3, verbose = TRUE)
fitARModel(best_params)
generateFeatures(best_params)
computeAccuracies(best_params)
```

## Results
- Results from experiments on UCR & UEA benchmark datasets demonstrate ARTSC's competitive performance.
- Example datasets include `Coffee`, `Haptics`, and `Lightning2`.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for feature suggestions or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


