# prepareDatasets.R

# Load the 'class' library for classification utilities
library(class)

# -----------------------------------------------------------------------------
# Helper function to load dataset file with error handling
# -----------------------------------------------------------------------------
load_dataset_file <- function(file_path) {
    tryCatch({
        # Check if the file exists before attempting to read
        if (!file.exists(file_path)) {
            stop(paste("Dataset file", file_path, "not found."))
        }
        # Read the file and convert it to a matrix
        as.matrix(read.table(file_path))
    }, error = function(e) {
        # Capture and re-throw any errors that occur during file loading
        stop(paste("Error loading dataset:", e$message))
    })
}

# -----------------------------------------------------------------------------
# Normalize data by standardizing each row
# (Subtract the mean and divide by the standard deviation)
# -----------------------------------------------------------------------------
normalize_data <- function(data, epsilon = 1e-6, verbose = FALSE) {
    t(apply(data, 1, function(x, i) {
        std_dev <- sd(x, na.rm = TRUE)
        
        # If standard deviation is zero or NaN, return a zero vector
        if (std_dev == 0 || is.na(std_dev)) {
            if (verbose) {
                warning(sprintf(
                    "Standard deviation is zero/NaN for instance %d. Returning zero vector.", 
                    i
                ))
            }
            numeric(length(x))  # Return zero vector of the same length
        } else {
            if (verbose) cat(sprintf("Normalizing instance %d\n", i))
            (x - mean(x, na.rm = TRUE)) / (std_dev + epsilon)
        }
    }, seq_len(nrow(data))))
}

# -----------------------------------------------------------------------------
# Imputation for missing values with additional methods (mean or median)
# -----------------------------------------------------------------------------
impute_missing_values <- function(data, method = "mean", verbose = FALSE) {
    # Determine the replacement value based on chosen method
    if (method == "mean") {
        replace_val <- mean(data, na.rm = TRUE)
    } else if (method == "median") {
        replace_val <- median(data, na.rm = TRUE)
    } else {
        stop("Unsupported imputation method. Use 'mean' or 'median'.")
    }
    
    # Replace all NA values with the computed replacement value
    data[is.na(data)] <- replace_val
    
    if (verbose) {
        cat(paste("Imputed missing values using", method, "\n"))
    }
    return(data)
}

# -----------------------------------------------------------------------------
# Prepare the dataset: load files, handle missing values, normalize if required,
# extract class labels, store metadata, and provide utility functions.
# -----------------------------------------------------------------------------
prepareDataset <- function(dataset_name,
                           dataset_folder = "Datasets/",
                           normalize = TRUE,
                           handle_na = TRUE,
                           impute_method = "mean",
                           verbose = FALSE) {
    
    # Validate dataset_name as a character string
    if (!is.character(dataset_name)) {
        stop("dataset_name must be a character string.")
    }
    
    # Initialize a list to hold all dataset information
    dataset <- list()
    
    if (verbose) {
        cat("Loading dataset:", dataset_name, "\n")
    }
    
    # Load train and test data from specified folder
    train_data_raw <- load_dataset_file(paste0(dataset_folder, dataset_name, "_TRAIN"))
    test_data_raw  <- load_dataset_file(paste0(dataset_folder, dataset_name, "_TEST"))
    
    if (verbose) {
        cat("Dataset loaded. Preparing data...\n")
    }
    
    # -----------------------------------------------------------------------------
    # Handle missing values if requested
    # -----------------------------------------------------------------------------
    if (handle_na) {
        # Check train data for missing values
        if (any(is.na(train_data_raw[, -1]))) {
            if (verbose) cat("Handling missing values in train data...\n")
            train_data_raw[, -1] <- impute_missing_values(
                train_data_raw[, -1],
                method = impute_method,
                verbose = verbose
            )
        }
        # Check test data for missing values
        if (any(is.na(test_data_raw[, -1]))) {
            if (verbose) cat("Handling missing values in test data...\n")
            test_data_raw[, -1] <- impute_missing_values(
                test_data_raw[, -1],
                method = impute_method,
                verbose = verbose
            )
        }
    } else if (any(is.na(train_data_raw[, -1])) || any(is.na(test_data_raw[, -1]))) {
        # Warn if user chose not to handle missing values but they exist
        warning("NA values found in the dataset. Please inspect the raw data.")
    }
    
    # Keep a copy of the original (unmodified) time series data
    dataset$train_timeseries_original <- train_data_raw[, -1]
    dataset$test_timeseries_original  <- test_data_raw[, -1]
    
    # -----------------------------------------------------------------------------
    # Normalize the data if requested
    # -----------------------------------------------------------------------------
    if (normalize) {
        train_data_raw[, -1] <- normalize_data(train_data_raw[, -1], verbose = verbose)
        test_data_raw[, -1]  <- normalize_data(test_data_raw[, -1], verbose = verbose)
    }
    
    # -----------------------------------------------------------------------------
    # Extract class labels (first column) and time series (remaining columns)
    # -----------------------------------------------------------------------------
    dataset$train_class      <- as.factor(train_data_raw[, 1])
    dataset$test_class       <- as.factor(test_data_raw[, 1])
    dataset$train_timeseries <- train_data_raw[, -1]
    dataset$test_timeseries  <- test_data_raw[, -1]
    
    # -----------------------------------------------------------------------------
    # Log metadata if verbose
    # -----------------------------------------------------------------------------
    if (verbose) {
        cat("Class labels and time series data extracted.\n")
        cat(sprintf(
            "Train size: %d, Test size: %d, Time series length: %d, Number of classes: %d\n",
            nrow(dataset$train_timeseries),
            nrow(dataset$test_timeseries),
            ncol(dataset$train_timeseries),
            length(unique(dataset$train_class))
        ))
    }
    
    # -----------------------------------------------------------------------------
    # Store additional metadata
    # -----------------------------------------------------------------------------
    dataset$time_series_length <- ncol(dataset$train_timeseries)
    dataset$number_of_train    <- nrow(dataset$train_timeseries)
    dataset$number_of_test     <- nrow(dataset$test_timeseries)
    dataset$class_values_list  <- unique(dataset$train_class)
    dataset$number_of_classes  <- length(dataset$class_values_list)
    
    # -----------------------------------------------------------------------------
    # Add utility functions to retrieve individual instances and plot them
    # -----------------------------------------------------------------------------
    
    # Get a single train time series instance by index
    dataset$get_train_timeseries_instance <- function(instance_number = 1) {
        dataset$train_timeseries[instance_number, ]
    }
    
    # Get a single test time series instance by index
    dataset$get_test_timeseries_instance <- function(instance_number = 1) {
        dataset$test_timeseries[instance_number, ]
    }
    
    # Plot a single train instance with customizable color and line type
    dataset$plot_train_instance <- function(instance_number = 1, col = "blue", lty = 1) {
        ts_ins <- dataset$get_train_timeseries_instance(instance_number)
        class_name <- dataset$train_class[instance_number]
        plot(
            ts_ins,
            main = paste("TRAIN:", dataset_name),
            sub  = paste("Instance:", instance_number, "Class:", class_name),
            col  = col,
            lty  = lty
        )
    }
    
    # Plot a single test instance with customizable color and line type
    dataset$plot_test_instance <- function(instance_number = 1, col = "red", lty = 1) {
        ts_ins <- dataset$get_test_timeseries_instance(instance_number)
        class_name <- dataset$test_class[instance_number]
        plot(
            ts_ins,
            main = paste("TEST:", dataset_name),
            sub  = paste("Instance:", instance_number, "Class:", class_name),
            col  = col,
            lty  = lty
        )
    }
    
    # Return the fully prepared dataset object
    return(dataset)
}
