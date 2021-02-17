#' Model error or loss
#' 
#' Mean squared or absolute error of a `wh` or `wdh` factorization. This measures how well a projection fits the model.
#'
#' Loss is not simply the sum of all sample losses when regularizations are applied. Regularization penalties are added to the loss.
#'
#' @inheritParams tlearn
#' @param d model diagonal (i.e. scaling matrix in NMF, singular values in SVD/PCA), or simply NULL or a vector of ones if no diagonal is included in the model. Default is NULL.
#' @param h sample coefficients model: matrix of factors x samples
#' @param error_method "mse" or "mae" for mean squared/absolute error
#' @param w_L1 L1/Lasso regularization on w
#' @param w_L2 L2/Ridge regression on w
#' @param w_PE PE/pattern extraction regularization on w
#' @param h_L1 L1/Lasso regularization on h
#' @param h_L2 L2/Ridge regression on h
#' @param h_PE PE/pattern extraction regularization on h
#' @param parallel parallelize error loss calculation, applies only if "A" is dense. Generally if "A" is more than 90 percent sparse, it is always faster to simply leave "A" as sparse and take advantage of the (unparallelized) sparse matrix iterator in the RcppArmadillo backend.
#' @returns mean squared/absolute error
#' @export
#'
error <- function(A, w, d = NULL, h, error_method = "mse", weights = NULL, h_L1 = 0, h_L2 = 0, h_PE = 0, w_L1 = 0, w_L2 = 0, w_PE = 0, parallel = TRUE) {
    
  if (nrow(w) != nrow(A)) {
    aw <- A_and_w_not_equal(A, w)
    A <- aw$A
    w <- aw$w
  }

  if (is.null(d)) d <- rep(1, ncol(w)) else if (class(d) == "matrix") d <- diag(d)
  if (is.null(weights)) weights <- as(matrix(1), "dgCMatrix")
  if (class(A) == "dgCMatrix") {
    return(Rcpp_SpLoss(as.matrix(w), as.vector(d), as.matrix(h), A, as(weights, "dgCMatrix"), as.double(w_L1), as.double(w_L2), as.double(w_PE), as.double(h_L1), as.double(h_L2), as.double(h_PE), ifelse(error_method == "mse", 1L, 0L)))
  } else return(Rcpp_Loss(as.matrix(w), as.vector(d), as.matrix(h), A, as.matrix(weights), as.double(w_L1), as.double(w_L2), as.double(w_PE), as.double(h_L1), as.double(h_L2), as.double(h_PE), ifelse(error_method == "mse", 1L, 0L), ifelse(parallel, 0L, 1L)))
}

#' Model error or loss for each sample
#'
#' Mean squared or absolute error for each sample in a factorization
#'
#' Useful to find how well samples fit (or do not fit) the model.
#'
#' @inheritParams error
#' @param relative return relative loss, the observed loss as a proportion of the squared sum (mse) or absolute sum (mae) of each sample
#' @param parallel parallelize error loss calculation
#' @returns numeric vector of loss for all samples
#' @export
#'
errorSample <- function(A, w, d = NULL, h, error_method = "mse", parallel = TRUE, relative = FALSE) {

  if (nrow(w) != nrow(A)) {
    aw <- A_and_w_not_equal(A, w)
    A <- aw$A
    w <- aw$w
  }

  if (is.null(d)) d <- rep(1, ncol(w)) else if (class(d) == "matrix") d <- diag(d)
  if (class(A) == "dgCMatrix") {
    loss_value <- Rcpp_SpLossSample(as.matrix(w), as.vector(d), as.matrix(h), A, ifelse(error_method == "mse", 1L, 0L), ifelse(parallel, 0L, 1L))
  } else {
    loss_value <- Rcpp_LossSample(as.matrix(w), as.vector(d), as.matrix(h), A, ifelse(error_method == "mse", 1L, 0L), ifelse(parallel, 0L, 1L))
  }
  if (relative) {
    ifelse(error_method == "mse", loss_value <- loss_value / colSums(A ^ 2 / ncol(A)), loss_value <- loss_value / colSums(abs(A) / ncol(A)))
  }
  return(sapply(loss_value, as.double))
}

A_and_w_not_equal <- function(A, w){

  # If "A" and "w" contain non-intersecting features, retain only those features which intersect
  if (nrow(w) != nrow(A)) {
    rw <- rownames(w)
    rA <- rownames(A)
    intersection <- intersect(rw, rA)
    if (length(intersection) > 0) {
      A <- A[which(rA %in% intersection),]
      w <- w[which(rw %in% intersection),]
      warning(paste("w and A did not have the same number of features. tlearn has successfully subset", length(intersection), "intersecting features and removed", (length(rA) + length(rw) - length(intersection)), "features in w and A using rownames"))
    } else stop("Number of features in w and A were not equal, and the features were either not named or there was no overlap in the names. Specify A and w with an equal number of features and/or ensure rownames overlap and intersect.")
  }

  return(list("A" = A, "w" = w))
}