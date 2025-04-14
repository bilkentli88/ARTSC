# findBestParameters.R
#
# This script identifies optimal AR model parameters (lag and differencing)
# by evaluating cross-validation accuracy. It uses:
#  - A single call to `prepareDataset(dataset_name)` to prepare the data.
#  - `generate_ar_based_features` (from generateFeatures.R) to produce AR-based features from the prepared data.
#  - A hybrid approach in `fitARModelFinal.R`, switching between ar() and arima().
#  - `computeAccuracies.R` for cross-validation and final train/test accuracies.

library(randomForest)  # For Random Forest model training
library(tseries)       # For KPSS test (stationarity check)

# -----------------------------------------------------------------------------
# Source necessary modules
# -----------------------------------------------------------------------------
source("generateFeatures.R")           # AR-based feature generation 
source("Helpers/prepareDatasets.R")  # Data preparation utilities 
source("Helpers/computeAccuracies.R")        # Cross-validation accuracy computation 
# source("Helpers/computeImportance.R")  # Optional for feature importance

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
set.seed(2019)  # For reproducibility
verbose <- TRUE

# -----------------------------------------------------------------------------
# User-specified parameters
# -----------------------------------------------------------------------------
dataset_name <- "GunPoint"  # Replace with the dataset name
method_name  <- "rf"         # Algorithm to use (e.g., 'rf', 'svm', 'knn')

# -----------------------------------------------------------------------------
# 1) Prepare the dataset
# -----------------------------------------------------------------------------
dataModel   <- prepareDataset(dataset_name)
num_classes <- dataModel$number_of_classes
L           <- dataModel$time_series_length

# -----------------------------------------------------------------------------
# 2) Compute complexity score and decide category
# -----------------------------------------------------------------------------
complexity_results  <- compute_complexity_score(dataModel$train_timeseries_original, num_classes)
complexity_score    <- complexity_results$complexity_score
complexity_category <- complexity_results$category  # "Simple", "Moderate", or "Complex"

if (verbose) {
    cat(sprintf("Processing dataset: %s\n", dataset_name))
    cat(sprintf(
        "Train size: %d, Test size: %d, Time series length: %d, Number of classes: %d\n",
        dataModel$number_of_train,
        dataModel$number_of_test,
        dataModel$time_series_length,
        num_classes
    ))
    cat(sprintf("Complexity Score: %.2f (%s)\n\n", complexity_score, complexity_category))
}

# -----------------------------------------------------------------------------
# 3) Set search ranges (p_max, d_max) based on complexity
# -----------------------------------------------------------------------------
if (complexity_category == "Simple") {
    p_max <- 24
    d_max <- 3
} else if (complexity_category == "Moderate") {
    p_max <- 30
    d_max <- 4
} else {
    p_max <- 36
    d_max <- 5
}

# -----------------------------------------------------------------------------
# 4) Initialize variables for tracking best parameters
# -----------------------------------------------------------------------------
best_cv_accuracy <- 0
best_lag_value   <- 0
best_diff_value  <- 0
best_model       <- NULL

p_value_threshold <- 0.01  # KPSS test threshold

# -----------------------------------------------------------------------------
# Helper function to check stationarity via KPSS
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
# 5) Main loop to search for the best (lag, diff) parameters
# -----------------------------------------------------------------------------
for (diff_value in seq(0, d_max)) {
    cat(sprintf("\nChecking diff_value: %d\n", diff_value))
    
    # Check stationarity for the first training instance
    if (!is_stationary_kpss(dataModel$train_timeseries_original[1, ], diff_value, p_value_threshold)) {
        cat(sprintf(
            "Series with diff_value = %d is not stationary, skipping to next diff_value...\n",
            diff_value
        ))
        next
    }
    
    for (lag_value in seq(2, p_max)) {
        tryCatch({
            # Generate AR-based features with the prepared dataModel using generateFeaturesOld
            m <- generate_ar_based_features(
                dataset_name  = dataset_name,
                LAG_VALUE     = lag_value,
                DIFF_VALUE    = diff_value,
                verbose       = FALSE
            )
            
            # Remove constant features from training and test sets
            df_train <- remove_constant_features(m$get_training_dataset())
            df_test  <- remove_constant_features(m$get_testing_dataset())
            
            if (is.null(df_train) || is.null(df_test)) {
                next
            }
            
            # Compute cross-validation accuracy
            accuracies  <- computeAccuraciesFinal(
                df_train,
                df_test,
                method_name = method_name,
                verbose     = FALSE
            )
            cv_accuracy <- accuracies$cv_accuracy
            
            cat(sprintf(
                "cv_accuracy for lag_value = %d, diff_value = %d: %.6f\n",
                lag_value, diff_value, cv_accuracy
            ))
            
            # Track the best parameters
            if (!is.na(cv_accuracy) && cv_accuracy > best_cv_accuracy) {
                best_cv_accuracy <- cv_accuracy
                best_lag_value   <- lag_value
                best_diff_value  <- diff_value
                best_model       <- m
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
# 6) Evaluate and save results with the best model
# -----------------------------------------------------------------------------
if (best_cv_accuracy == 0) {
    cat("No valid stationary series found for any combination of p and d.\n")
} else {
    cat(sprintf(
        "\nBest parameters for %s: best_lag_value = %d, best_diff_value = %d, best_cv_accuracy = %.5f\n",
        dataset_name, best_lag_value, best_diff_value, best_cv_accuracy
    ))
    
    # Retrieve final train and test sets
    df_train <- best_model$get_training_dataset()
    df_test  <- best_model$get_testing_dataset()
    
    # Compute final train/test accuracies
    accuracies <- computeAccuraciesFinal(df_train, df_test, method_name = method_name, verbose = FALSE)
    cat(sprintf("Training accuracy: %.4f\n", accuracies$train_accuracy))
    cat(sprintf("Test accuracy: %.4f\n", accuracies$test_accuracy))
    
    # Save results
    if (!dir.exists("ARTSCResults")) {
        dir.create("ARTSCResults")
    }
    file_name <- paste0(
        "ARTSCResults/", dataset_name, "_results.csv"
    )
    
    results_df <- data.frame(
        dataset_name   = dataset_name,
        lag_value      = best_lag_value,
        diff_value     = best_diff_value,
        train_accuracy = accuracies$train_accuracy,
        test_accuracy  = accuracies$test_accuracy
    )
    # Optionally call feature importance (commented out)
    # feature_importance <- compute_feature_importance(df_train, dataset_name)
    
    write.csv(results_df, file = file_name, row.names = FALSE)
    cat(paste("Results have been saved to", file_name, "\n"))
}
