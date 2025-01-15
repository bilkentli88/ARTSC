# computeAccuraciesFinal.R

library(caret)
library(dplyr)
library(randomForest)

# -----------------------------------------------------------------------------
# Helper function to preprocess the data (OLD LOGIC)
# - Remove constant features
# - Remove any rows with NA values (na.omit)
# - Stop if no features remain
# -----------------------------------------------------------------------------
preprocess_data <- function(df) {
    non_constant_cols <- sapply(df, function(x) length(unique(x)) > 1)
    df <- df[, non_constant_cols, drop = FALSE]
    
    df <- na.omit(df)  # OLD approach: remove rows containing NA
    
    if (nrow(df) == 0 || ncol(df) <= 1) {
        stop("Insufficient features in dataset after preprocessing.")
    }
    return(df)
}

# -----------------------------------------------------------------------------
# Helper function to compute complexity score
# (same logic as both old/new for categorizing: Simple, Moderate, Complex)
# -----------------------------------------------------------------------------
compute_complexity_score <- function(df_train, num_classes) {
    P <- ncol(df_train) - 1
    N <- nrow(df_train)
    C <- num_classes
    
    complexity_score <- (P * C) / log(N + 1)
    category <- if (complexity_score < 120) {
        "Simple"
    } else if (complexity_score <= 500) {
        "Moderate"
    } else {
        "Complex"
    }
    
    return(list(complexity_score = complexity_score, category = category))
}

# -----------------------------------------------------------------------------
# Main function to compute accuracies using cross-validation
# -----------------------------------------------------------------------------
computeAccuraciesFinal <- function(df_train,
                                   df_test,
                                   method_name = "rf",
                                   cv_count    = 10,
                                   verbose     = FALSE) {
    # Use the OLD preprocessing logic (with na.omit)
    df_train <- preprocess_data(df_train)
    df_test  <- preprocess_data(df_test)
    
    if (nrow(df_train) == 0 || nrow(df_test) == 0) {
        stop("No valid data available after preprocessing.")
    }
    
    # Determine dataset complexity
    num_classes     <- length(unique(df_train$class))
    complexity_info <- compute_complexity_score(df_train, num_classes)
    
    if (verbose) {
        cat(sprintf(
            "Complexity score: %.2f, Category: %s\n",
            complexity_info$complexity_score,
            complexity_info$category
        ))
    }
    
    # -----------------------------------------------------------------------------
    # EXACT "Simple" settings from old module, "Moderate"/"Complex" from newer code
    # -----------------------------------------------------------------------------
    rf_params <- switch(
        complexity_info$category,
        "Simple" = list(
            # OLD approach: keep trees unconstrained, specify mtry & nodesize, maxnodes=NULL
            nodesize = 3,
            maxnodes = NULL,
            mtry     = floor(sqrt(ncol(df_train) - 1))
            # ntree is not specified => default ~ 500
        ),
        "Moderate" = list(
            # New approach: fix ntree=200, nodesize=2, do not specify mtry => caret search
            ntree    = 200,
            nodesize = 2
        ),
        "Complex" = list(
            # New approach: fix ntree=500, nodesize=1, do not specify mtry => caret search
            ntree    = 500,
            nodesize = 1
        )
    )
    
    # -----------------------------------------------------------------------------
    # Cross-validation accuracy computation
    # -----------------------------------------------------------------------------
    cv_accuracy <- tryCatch({
        train_control <- trainControl(method = "cv", number = cv_count, allowParallel = TRUE)
        
        if (method_name == "rf") {
            
            if (complexity_info$category == "Simple") {
                # EXACT old approach for "Simple":
                # - Single mtry in tuneGrid
                # - nodesize & maxnodes => randomForest arguments
                # - no ntree => default ~500
                cv_model <- train(
                    class ~ .,
                    data      = df_train,
                    method    = method_name,
                    trControl = train_control,
                    tuneGrid  = data.frame(mtry = rf_params$mtry),
                    nodesize  = rf_params$nodesize,
                    maxnodes  = rf_params$maxnodes
                    # no ntree -> default is typically 500 in randomForest
                )
                
            } else if (complexity_info$category == "Moderate") {
                # Use new approach: fix ntree=200, nodesize=2, omit mtry => caret's default search
                cv_model <- train(
                    class ~ .,
                    data      = df_train,
                    method    = method_name,
                    trControl = train_control,
                    ntree     = rf_params$ntree,
                    nodesize  = rf_params$nodesize
                )
                
            } else {
                # "Complex": ntree=500, nodesize=1
                cv_model <- train(
                    class ~ .,
                    data      = df_train,
                    method    = method_name,
                    trControl = train_control,
                    ntree     = rf_params$ntree,
                    nodesize  = rf_params$nodesize
                )
            }
            
        } else {
            # If the chosen method is not "rf", just train normally
            cv_model <- train(
                class ~ .,
                data      = df_train,
                method    = method_name,
                trControl = train_control
            )
        }
        
        # Return the mean CV accuracy
        mean(cv_model$results$Accuracy)
        
    }, error = function(e) {
        warning(sprintf("Error during cross-validation: %s", e$message))
        NA
    })
    
    # -----------------------------------------------------------------------------
    # Training accuracy
    # -----------------------------------------------------------------------------
    find_train_accuracy <- function(df) {
        tryCatch({
            if (complexity_info$category == "Simple") {
                # EXACT old approach for "Simple"
                rf_model <- randomForest(
                    class ~ .,
                    data     = df,
                    mtry     = rf_params$mtry,
                    nodesize = rf_params$nodesize,
                    maxnodes = rf_params$maxnodes
                    # ntree => default 500
                )
                
            } else if (complexity_info$category == "Moderate") {
                rf_model <- randomForest(
                    class ~ .,
                    data     = df,
                    ntree    = rf_params$ntree,
                    nodesize = rf_params$nodesize
                )
                
            } else {
                rf_model <- randomForest(
                    class ~ .,
                    data     = df,
                    ntree    = rf_params$ntree,
                    nodesize = rf_params$nodesize
                )
            }
            
            preds <- predict(rf_model, df)
            mean(preds == df$class)
            
        }, error = function(e) {
            warning(sprintf("Error calculating training accuracy: %s", e$message))
            return(NA)
        })
    }
    
    # -----------------------------------------------------------------------------
    # Test accuracy
    # -----------------------------------------------------------------------------
    find_test_accuracy <- function(train_df, test_df) {
        tryCatch({
            if (complexity_info$category == "Simple") {
                # EXACT old approach for "Simple"
                rf_model <- randomForest(
                    class ~ .,
                    data     = train_df,
                    mtry     = rf_params$mtry,
                    nodesize = rf_params$nodesize,
                    maxnodes = rf_params$maxnodes
                )
                
            } else if (complexity_info$category == "Moderate") {
                rf_model <- randomForest(
                    class ~ .,
                    data     = train_df,
                    ntree    = rf_params$ntree,
                    nodesize = rf_params$nodesize
                )
                
            } else {
                rf_model <- randomForest(
                    class ~ .,
                    data     = train_df,
                    ntree    = rf_params$ntree,
                    nodesize = rf_params$nodesize
                )
            }
            
            preds <- predict(rf_model, test_df)
            mean(preds == test_df$class)
            
        }, error = function(e) {
            warning(sprintf("Error calculating test accuracy: %s", e$message))
            return(NA)
        })
    }
    
    # -----------------------------------------------------------------------------
    # Compute final accuracies
    # -----------------------------------------------------------------------------
    train_accuracy <- find_train_accuracy(df_train)
    test_accuracy  <- find_test_accuracy(df_train, df_test)
    
    # -----------------------------------------------------------------------------
    # Return results
    # -----------------------------------------------------------------------------
    return(list(
        cv_accuracy    = cv_accuracy,
        train_accuracy = train_accuracy,
        test_accuracy  = test_accuracy
    ))
}
