#' Find the best binary value for a projection
#'
#' Finds the best zero/non-zero value pair that minimizes the loss for a projection and returns "h" at that value
#'
#' Given a binary projection on a multinomial distribution comprised of zero and a real non-negative number, the value of that non-negative number must be optimized to minimize the loss of the projection.
#'
#' The relationship between that value and the loss of the projection is perfectly second-order and polynomial.
#' Thus, `tlearn_binary` begins by calculating loss at four values, then fits a quadratic fit in the form `y = ax^2 + bx + c`, and finds the predicted minima at `-b/(2a)`.
#' `tlearn_binary` then projects the model at the predicted minima, calculates the loss of the projection, and measures the relative error of the predicted loss based on the quadratic fit against the observed loss.
#' The process is repeated until the relative error (stopping tolerance for convergence) falls below `tol`.
#'
#' @inheritParams tlearn
#' @param error_method "mse" or "mae" for mean squared/absolute error
#' @param parallel parallelize error loss calculation, applies only if "A" is dense. Generally if "A" is more than 90 percent sparse, it is always faster to simply leave "A" as sparse and take advantage of the (unparallelized) sparse matrix iterator in the RcppArmadillo backend.
#' @param verbose show convergence values
#' @param start.values vector of at least four binary values which will be used to initialize a quadratic fit of loss vs. binary value (see details). These values should include an underestimate, two good guesses on either side of the optimal binary value, and an overestimate.
#' @param fit_tol convergence tolerance given as the relative accuracy of the quadratic fit at the minima of the objective (see details). Default should satisfy.
#' @param ... any parameter supported by `tlearn`
#' @returns "h" for the projection at the best binary value
#' @importFrom stats lm
#' @export
#'
tlearn_binary <- function(A, w, start.values = c(0.5, 0.7, 0.9, 1.1), verbose = TRUE, fit_tol = 0.01, error_method = "mse", parallel = TRUE, ...) {

  # find 4 values to fit the model
  losses <- list()
  if(verbose) message("finding best binary value by quadratic fitting and minima searching:\n")
  if (verbose) message("   value      | loss\n  -------------------------")
  for (i in 1:length(start.values)) {
    if (verbose) message("   ", formatC(start.values[i], format = "e"), " | NA, start val")
    h <- tlearn(A, w, values = c(0, start.values[i]), ...)
    losses[[i]] <- c("value" = start.values[i], "loss" =  error(A, w, h = h, parallel = parallel, error_method = error_method))
  }
  df <- data.frame(do.call(rbind, losses))
  minima_tol <- 1

  while (minima_tol > fit_tol) {
    # fit a second-order polynomial model to estiamte the minimum
    fit <- lm(df$loss ~ df$value + I(df$value ^ 2))

    # estimate minimum. For a model fit in the form y = ax^2 + bx + c, the minima is given by -b/2a
    minima <- as.double(-fit$coefficients[2] / (2 * fit$coefficients[3]))

    # project at the minima, calculate tolerance as the predicted loss vs the actual loss abs(predicted - actual) / actual
    h <- tlearn(A, w, values = c(0, minima), ...)

    minima_loss <- error(A, w, h = h, parallel = parallel, error_method = error_method)
    minima_tol <- abs(minima_loss - (fit$coefficients[3] * minima ^ 2 + fit$coefficients[2] * minima + fit$coefficients[1])) / minima_loss
    losses[[length(losses) + 1]] <- c("value" = minima, "loss" = minima_loss)
    df <- data.frame(do.call(rbind, losses))
    if (verbose) message("   ", formatC(minima, format = "e"), " | ", formatC(minima_tol, format = "e"))
  }

  return(h)
}