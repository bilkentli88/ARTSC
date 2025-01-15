# fitARModel.R

# This script provides AR model utilities, including:
#  - Differencing time series instances.
#  - Fitting an AR model with dynamically adjusted lag, using:
#       -- ar() if complexity_category == "Simple"
#       -- arima(..., order=(lag,0,0)) otherwise
#  - Estimating values from AR coefficients.
#  - Computing error between actual and estimated time series.

# -----------------------------------------------------------------------------
# Function to compute the differenced time series
# -----------------------------------------------------------------------------
find_differenced_ts_instance <- function(ts_instance, diff_value) {
    if (diff_value < 0) {
        stop("diff_value cannot be negative.")
    }
    if (diff_value > 0) {
        return(diff(ts_instance, differences = diff_value))
    } else {
        return(ts_instance)
    }
}

# -----------------------------------------------------------------------------
# Extract AR parameters (lag and differencing) with error handling,
# using a hybrid approach based on complexity_category:
#   - "Simple"   => use ar()
#   - otherwise  => use arima(..., order=(lag,0,0))
# -----------------------------------------------------------------------------
find_ar_param_values <- function(current_ts,
                                 LAG_VALUE           = 8,
                                 DIFF_VALUE          = 0,
                                 min_lag_value       = 2,
                                 complexity_category = "Simple") {
    
    # Apply differencing if needed
    differenced_ts <- current_ts
    if (DIFF_VALUE > 0) {
        if (length(current_ts) <= DIFF_VALUE) {
            warning(sprintf(
                "Cannot difference time series of length %d with diff_value = %d",
                length(current_ts), DIFF_VALUE
            ))
            return(NULL)
        }
        differenced_ts <- diff(current_ts, differences = DIFF_VALUE)
    }
    
    # Dynamically adjust lag based on the differenced series length
    if (length(differenced_ts) <= LAG_VALUE) {
        adjusted_lag <- max(min_lag_value, floor(length(differenced_ts) - 1))
        warning(sprintf(
            "Adjusting lag_value from %d to %d for series of length %d",
            LAG_VALUE, adjusted_lag, length(differenced_ts)
        ))
        LAG_VALUE <- adjusted_lag
    } else {
        adjusted_lag <- LAG_VALUE
    }
    
    # Attempt to fit an AR model with the adjusted lag
    tryCatch({
        if (complexity_category == "Simple") {
            # Use ar() for "Simple" datasets
            ar_model_fit <- ar(
                x         = differenced_ts,
                order.max = adjusted_lag,
                aic       = FALSE,
                method    = "yule-walker"   # or "mle"/"burg"/"ols"
            )
            coef_vals <- ar_model_fit$ar
        } else {
            # Use arima(..., order=(adjusted_lag, 0, 0)) for Moderate/Complex
            ar_model_fit <- arima(
                x      = differenced_ts,
                order  = c(adjusted_lag, 0, 0),
                method = "CSS"
            )
            coef_vals <- ar_model_fit$coef
        }
        
        return(list(
            lag_value    = adjusted_lag,
            diff_value   = DIFF_VALUE,
            coefficients = coef_vals,
            model        = ar_model_fit
        ))
    }, error = function(e) {
        warning(sprintf(
            "AR model fitting failed for lag = %d, diff = %d on series of length %d. Error: %s",
            adjusted_lag, DIFF_VALUE, length(differenced_ts), e$message
        ))
        return(NULL)
    })
}

# -----------------------------------------------------------------------------
# Estimate values based on AR coefficients
# -----------------------------------------------------------------------------
find_estimated_values_for_one_instance <- function(differenced_ts_instance,
                                                   lag_value,
                                                   coef_values) {
    n <- length(differenced_ts_instance)
    
    if (n < lag_value) {
        stop(sprintf(
            "Time series length (%d) is shorter than the specified lag_value (%d).",
            n, lag_value
        ))
    }
    
    ts_estimated <- rep(0, n)
    ts_estimated[1:lag_value] <- differenced_ts_instance[1:lag_value]
    
    for (i in (lag_value + 1):n) {
        ts_estimated[i] <- sum(
            differenced_ts_instance[(i - lag_value):(i - 1)] *
                coef_values[1:lag_value]
        )
    }
    
    return(ts_estimated)
}

# -----------------------------------------------------------------------------
# Compute error between actual and estimated time series
# -----------------------------------------------------------------------------
get_error_value_for_one_instance <- function(ts_instance,
                                             lag_value,
                                             diff_value,
                                             coef_values,
                                             cached_differenced_ts = NULL,
                                             verbose = FALSE) {
    
    # Use cached differenced series if provided, otherwise compute it
    if (is.null(cached_differenced_ts)) {
        differenced_ts_instance <- find_differenced_ts_instance(ts_instance, diff_value)
    } else {
        differenced_ts_instance <- cached_differenced_ts
    }
    
    # Adjust lag if necessary before estimating values
    adjusted_lag <- min(lag_value, length(differenced_ts_instance))
    ts_estimated <- find_estimated_values_for_one_instance(
        differenced_ts_instance,
        adjusted_lag,
        coef_values
    )
    
    if (verbose) {
        cat(sprintf(
            "Error value for instance computed with lag = %d, diff_value = %d\n",
            lag_value, diff_value
        ))
    }
    
    error <- (differenced_ts_instance - ts_estimated)^2
    return(sum(error))
}
