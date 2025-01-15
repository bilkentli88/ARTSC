# generateFeaturesOld.R

# Load necessary helper modules
source("Helpers/prepareDatasets.R")
source("Helpers/fitARModel.R")

generate_ar_based_features <- function(dataset_name, LAG_VALUE, DIFF_VALUE, verbose = FALSE) {
    
    create_dataset <- function(timeseries_data, class_data, num_instances, dataset_type) {
        list_for_df <- list()
        max_lag_value <- 0
        
        for (instance_index in seq(num_instances)) {
            current_ts <- timeseries_data[instance_index, ]
            current_class <- class_data[instance_index]
            
            differenced_ts <- current_ts
            
            if (DIFF_VALUE > 0 && length(current_ts) > DIFF_VALUE) {
                differenced_ts <- diff(current_ts, differences = DIFF_VALUE)
            }
            
            ar_results <- tryCatch({
                ar_model <- ar(differenced_ts, order.max = LAG_VALUE, aic = FALSE)
                list(lag_value = length(ar_model$ar), diff_value = DIFF_VALUE, coefficients = ar_model$ar)
            }, error = function(e) {
                if (verbose) warning(sprintf("Error in AR fitting for %s instance %d: %s", dataset_type, instance_index, e$message))
                return(NULL)
            })
            
            if (is.null(ar_results)) next
            
            error_value <- get_error_value_for_one_instance(differenced_ts, ar_results$lag_value, ar_results$diff_value, ar_results$coefficients)
            
            instance_data <- list(
                class = current_class,
                lag_value = ar_results$lag_value,
                diff_value = ar_results$diff_value,
                coefficients = ar_results$coefficients,
                t_values = current_ts[1:ar_results$lag_value],
                error_value = error_value
            )
            
            if (instance_data$lag_value > max_lag_value) max_lag_value <- instance_data$lag_value
            list_for_df[[instance_index]] <- instance_data
            
            if (verbose) cat(sprintf("Processed %s instance %d with lag = %d.\n", dataset_type, instance_index, instance_data$lag_value))
        }
        
        df <- data.frame(
            class = sapply(list_for_df, function(x) x$class),
            lag_value = sapply(list_for_df, function(x) x$lag_value),
            diff_value = sapply(list_for_df, function(x) x$diff_value),
            error_value = sapply(list_for_df, function(x) x$error_value)
        )
        
        for (i in 1:max_lag_value) {
            col_name <- paste0("c", i)
            df[[col_name]] <- sapply(list_for_df, function(x) {
                if (i <= length(x$coefficients)) x$coefficients[i] else NA
            })
        }
        
        for (i in 1:max_lag_value) {
            col_name <- paste0("t", i)
            df[[col_name]] <- sapply(list_for_df, function(x) {
                if (i <= length(x$t_values)) x$t_values[i] else NA
            })
        }
        return(df)
    }
    
    dataModel <- prepareDataset(dataset_name)
    
    get_training_dataset <- function() {
        create_dataset(dataModel$train_timeseries_original, dataModel$train_class, dataModel$number_of_train, "training")
    }
    
    get_testing_dataset <- function() {
        create_dataset(dataModel$test_timeseries_original, dataModel$test_class, dataModel$number_of_test, "testing")
    }
    
    dataModel$get_training_dataset <- get_training_dataset
    dataModel$get_testing_dataset <- get_testing_dataset
    
    return(dataModel)
}
