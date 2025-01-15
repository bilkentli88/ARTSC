library(randomForest)

# Define how many top features to show in verbose mode (e.g., top 20 or more)
n_features <- 30

remove_constant_features <- function(df, verbose = FALSE) {
    # Identify columns with constant values
    non_constant_cols <- sapply(df, function(x) length(unique(x)) > 1)
    removed_count <- sum(!non_constant_cols)
    
    # Log removal if verbose mode is on
    if (verbose && removed_count > 0) {
        cat(sprintf("Removed %d constant columns. Remaining columns: %d\n", removed_count, sum(non_constant_cols)))
    }
    
    return(df[, non_constant_cols])
}

# Helper to detect detrimental features with their impact score
detect_detrimental_features <- function(importance_df, threshold = 0) {
    detrimental_features <- importance_df[importance_df$MeanDecreaseAccuracy < threshold, ]
    return(detrimental_features)
}

# Enhanced Feature Importance Plot with Gradient
plot_feature_importance <- function(rf_model, dataset_name, top_n = 30) {
    # Extract and sort feature importance
    importance_vals <- importance(rf_model)
    importance_df <- as.data.frame(importance_vals)
    importance_df$Feature <- rownames(importance_df)
    importance_df <- importance_df[order(-importance_df$MeanDecreaseAccuracy), ]  # Sort by importance
    
    # Select top features
    top_features <- importance_df[1:top_n, ]
    
    # Color gradient for bars (Blue -> Green -> Yellow)
    bar_colors <- colorRampPalette(c("blue", "violet", "purple"))(top_n)
    
    # Create a horizontal bar plot
    par(mar = c(5, 10, 4, 2))  # Adjust margins for better visibility
    barplot(
        height = top_features$MeanDecreaseAccuracy,
        names.arg = top_features$Feature,
        horiz = TRUE,
        las = 1,  # Make labels horizontal
        col = bar_colors,
        main = paste("Top 30 Most Important Features of", dataset_name),
        xlab = "Mean Decrease in Accuracy",
        cex.names = 0.8  # Adjust label size
    )
}

# Function to compute and save feature importance
compute_feature_importance <- function(df_train, dataset_name, save_plot = TRUE, verbose = FALSE, threshold = 0) {
    # Remove constant features
    df_train <- remove_constant_features(df_train, verbose = verbose)
    
    if (ncol(df_train) == 0) {
        stop("All features in the dataset are constant. Cannot compute feature importance.")
    }
    
    # Train Random Forest model
    rf_model <- tryCatch({
        randomForest(class ~ ., data = df_train, importance = TRUE)
    }, error = function(e) {
        warning(paste("Error while training Random Forest model for feature importance on dataset", dataset_name, ":", e$message))
        return(NULL)
    })
    
    # If the model failed, return NULL
    if (is.null(rf_model)) {
        return(NULL)
    }
    
    # Compute importance and retain relevant columns
    importance_vals <- importance(rf_model)
    importance_df <- as.data.frame(importance_vals)
    
    # Add the "Feature" column to retain feature names
    importance_df$Feature <- rownames(importance_df)
    importance_df <- importance_df[, c("Feature", "MeanDecreaseAccuracy", "MeanDecreaseGini")]  # Select only necessary columns
    
    # Detect detrimental features
    detrimental_features <- detect_detrimental_features(importance_df, threshold)
    
    # Save the importance plot as a figure
    if (save_plot) {
        if (!dir.exists("Figures")) dir.create("Figures")
        png(paste0("Figures/", dataset_name, "_FeatureImportance.png"))
        plot_feature_importance(rf_model, dataset_name)
        dev.off()
        
        if (verbose) {
            cat(paste("Feature importance plot saved to Figures/", dataset_name, "_FeatureImportance.png\n"))
        }
    }
    
    # Save the filtered feature importance to CSV
    if (!dir.exists("ARTSCResults")) dir.create("ARTSCResults")
    csv_filename <- paste0("ARTSCResults/", dataset_name, "_FeatureImportance_.csv")
    write.csv(importance_df, file = csv_filename, row.names = FALSE)
    
    return(list(importance_df = importance_df, detrimental_features = detrimental_features))
}
