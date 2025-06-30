# findBestParameters.R
#
# Identifies optimal AR model parameters (lag and differencing) for the ARTSC
# method by maximizing cross-validation accuracy. Uses a hybrid AR/ARIMA approach
# and evaluates datasets from the UCR Time Series Classification Archive.
#
# Dependencies:
# - generateFeatures.R: For generate_ar_based_features()
# - Helpers/prepareDatasets.R: For prepareDataset()
# - Helpers/computeAccuracies.R: For computeAccuraciesFinal()
# - randomForest: For Random Forest model training
# - tseries: For KPSS test (stationarity check)
#
# Outputs results to ARTSCResults/<dataset_name>_results_seed_<seed>.csv.
#
# Author: Aykut T. Altay
# Date: Jan 2025

library(randomForest)  # For Random Forest model training
library(tseries)      # For KPSS test (stationarity check)

# -----------------------------------------------------------------------------
# Source necessary modules
# -----------------------------------------------------------------------------
source("generateFeatures.R")          # AR-based feature generation
source("Helpers/prepareDatasets.R")   # Data preparation utilities
source("Helpers/computeAccuracies.R") # Cross-validation accuracy computation
# source("Helpers/computeImportance.R") # Optional: feature importance

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
seed_value <- 375   # Define seed for reproducibility
set.seed(seed_value)      # Ensure reproducible results
verbose <- TRUE           # Enable verbose output
p_value_threshold <- 0.01 # KPSS test threshold

# -----------------------------------------------------------------------------
# User-specified parameters
# -----------------------------------------------------------------------------
dataset_name <- "FordB" # Dataset name (e.g., "OsuLeaf", "ECG200")
method_name <- "rf"       # Algorithm: "rf", "svm", or "knn"

# -----------------------------------------------------------------------------
# 1) Load and prepare dataset
# -----------------------------------------------------------------------------
dataModel <- prepareDataset(dataset_name)
num_classes <- dataModel$number_of_classes
L <- dataModel$time_series_length

# -----------------------------------------------------------------------------
# 2) Compute complexity score and category
# -----------------------------------------------------------------------------
complexity_results <- compute_complexity_score(
    dataModel$train_timeseries_original,
    num_classes
)
complexity_score <- complexity_results$complexity_score
complexity_category <- complexity_results$category # Simple, Moderate, or Complex

if (verbose) {
    cat(sprintf("Processing dataset: %s\n", dataset_name))
    cat(sprintf(
        "Train size: %d, Test size: %d, Time series length: %d, Classes: %d\n",
        dataModel$number_of_train,
        dataModel$number_of_test,
        L,
        num_classes
    ))
    cat(sprintf("Complexity Score: %.2f (%s)\n\n", complexity_score, complexity_category))
}

# -----------------------------------------------------------------------------
# 3) Set search ranges based on complexity
# -----------------------------------------------------------------------------
if (complexity_category == "Simple") {
    p_max <- 25
    d_max <- 3
} else if (complexity_category == "Moderate") {
    p_max <- 29
    d_max <- 4
} else {
    p_max <- 45
    d_max <- 6
}

# -----------------------------------------------------------------------------
# 4) Initialize variables for tracking best parameters
# -----------------------------------------------------------------------------
best_cv_accuracy <- 0
best_lag_value <- 0
best_diff_value <- 0
best_model <- NULL

# -----------------------------------------------------------------------------
# Helper function: Check stationarity via KPSS test
# -----------------------------------------------------------------------------
is_stationary_kpss <- function(ts, diff_value, p_value_threshold) {
    if (diff_value > 0) {
        ts <- diff(ts, differences = diff_value)
    }
    kpss_test <- tryCatch({
        suppressWarnings(kpss.test(ts))
    }, error = function(e) {
        warning(sprintf(
            "KPSS test failed for diff_value = %d: %s",
            diff_value, e$message
        ))
        return(NULL)
    })
    if (is.null(kpss_test)) {
        return(FALSE)
    } else {
        return(kpss_test$p.value >= p_value_threshold)
    }
}

# -----------------------------------------------------------------------------
# 5) Main loop: Search for best (lag, diff) parameters
# -----------------------------------------------------------------------------
for (diff_value in seq(0, d_max)) {
    cat(sprintf("\nChecking diff_value: %d\n", diff_value))
    
    # Check stationarity for first training instance
    if (!is_stationary_kpss(dataModel$train_timeseries_original[1, ], diff_value, p_value_threshold)) {
        cat(sprintf(
            "Series with diff_value = %d is not stationary, skipping...\n",
            diff_value
        ))
        next
    }
    
    for (lag_value in seq(2, p_max)) {
        tryCatch({
            # Generate AR-based features
            m <- generate_ar_based_features(
                dataset_name = dataset_name,
                LAG_VALUE = lag_value,
                DIFF_VALUE = diff_value,
                verbose = FALSE
            )
            
            # Remove constant features
            df_train <- preprocess_data(m$get_training_dataset())
            df_test <- preprocess_data(m$get_testing_dataset())
            
            if (is.null(df_train) || is.null(df_test)) {
                next
            }
            
            # Compute cross-validation accuracy
            accuracies <- computeAccuraciesFinal(
                df_train,
                df_test,
                method_name = method_name,
                verbose = FALSE
            )
            cv_accuracy <- accuracies$cv_accuracy
            
            cat(sprintf(
                "CV accuracy for lag_value = %d, diff_value = %d: %.6f\n",
                lag_value, diff_value, cv_accuracy
            ))
            
            # Track best parameters
            if (!is.na(cv_accuracy) && cv_accuracy > best_cv_accuracy) {
                best_cv_accuracy <- cv_accuracy
                best_lag_value <- lag_value
                best_diff_value <- diff_value
                best_model <- m
            }
        }, error = function(e) {
            warning(sprintf(
                "Error in AR fitting for dataset: %s, diff_value: %d, lag_value: %d. Error: %s",
                dataset_name, diff_value, lag_value, e$message
            ))
        })
    }
}

# -----------------------------------------------------------------------------
# 6) Evaluate and save results with best model
# -----------------------------------------------------------------------------
if (best_cv_accuracy == 0) {
    cat("No valid stationary series found for any combination of p and d.\n")
} else {
    cat(sprintf(
        "\nBest parameters for %s: lag_value = %d, diff_value = %d, cv_accuracy = %.5f\n",
        dataset_name, best_lag_value, best_diff_value, best_cv_accuracy
    ))
    
    # Retrieve final train and test sets
    df_train <- best_model$get_training_dataset()
    df_test <- best_model$get_testing_dataset()
    
    # Compute final accuracies
    accuracies <- computeAccuraciesFinal(
        df_train,
        df_test,
        method_name = method_name,
        verbose = FALSE
    )
    cat(sprintf("Training accuracy: %.4f\n", accuracies$train_accuracy))
    cat(sprintf("Test accuracy: %.4f\n", accuracies$test_accuracy))
    
    # Save results
    if (!dir.exists("ARTSCResults")) {
        dir.create("ARTSCResults")
    }
    file_name <- paste0(
        "ARTSCResults/", dataset_name, "_results_seed_", seed_value, ".csv"
    )
    
    results_df <- data.frame(
        dataset_name = dataset_name,
        lag_value = best_lag_value,
        diff_value = best_diff_value,
        train_accuracy = accuracies$train_accuracy,
        test_accuracy = accuracies$test_accuracy,
        seed = seed_value
    )
    
    # Optionally compute feature importance (commented out)
    # feature_importance <- compute_feature_importance(df_train, dataset_name)
    
    write.csv(results_df, file = file_name, row.names = FALSE)
    cat(sprintf("Results saved to %s\n", file_name))
}
