#' Transfer learn
#'
#' Project samples (A) onto a factor model (w) to find the best sample embedding (h)
#'
#' The factor model may be any dimensional reduction (commonly NMF, PCA, or SVD).
#'
#' If "A" and "w" contain non-intersecting features, only features which intersect will be retained.
#'
#' If you'd like to monitor the progress of a very large projection and do not need weights, regularizations, or a warm start initialization, see \code{\link{tlearn_minibatch}}
#'
#' @param A samples to project: matrix of features x samples, dense or sparse
#' @param w feature coefficients model: matrix of features x factors
#' @param h.init initial sample coefficients model: optional matrix of factors x samples (leave NULL for random initialization)
#' @param weights optional weights/maskings given as a vector of ncol(A), nrow(A), or a dense or sparse matrix of dim(A). Leave NULL for no weighting
#' @param min minimum permitted value, zero for NMF
#' @param max maximum permitted value, default 1e10
#' @param values optional, constrain projection values to a multinomial distribution containing these values
#' @param tol tolerance for convergence of the least squares sequential coordinate descent solver
#' @param maxit maximum number of iterations for least squares sequential coordinate descent
#' @param parallel use all available CPU threads
#' @param L0 L0 truncation on each sample in "h", the number of non-zero values in each column. Must be less than the number of columns in "w" (rank of the model)
#' @param L1 L1/Lasso regularization to be applied on "h"
#' @param L2 L2/Ridge regression to be applied on "h"
#' @param PE pattern extraction penalty to be applied on "h" (i.e. force patterns of weightings to be different across rows of "h")
#' @param parallel logical, use all available threads
#' @returns h, a sample embeddings matrix of dimensions "factors x samples".
#' @seealso \code{\link{tlearn_minibatch}}
#' @export
#'
tlearn <- function(A, w, h.init = NULL, L0 = ncol(w), L1 = 0, L2 = 0, PE = 0, weights = NULL, min = 0, max = 1e10, values = NULL, parallel = TRUE, maxit = 50, tol = 1e-8) {

  # if values are set to NULL, make it an array of zeros (this is interpreted by Rcpp code as no action)
  if(is.null(values)) values <- c(0, 0)

  # If "A" and "w" contain non-intersecting features, retain only those features which intersect
  if (nrow(w) != nrow(A)) {
    aw <- A_and_w_not_equal(A, w)
    A <- aw$A
    w <- aw$w
  }

  if (!is.null(h.init)) {
    if (nrow(h.init) != ncol(w) || ncol(h.init) != ncol(A)) {
      h.init <- NULL
      warning("h.init was specified, but the number of rows was not equal to the number of columns in \"w\", and the number of columns was not equal to the number of columns in \"A\". An h.init matrix will be generated automatically with proper dimensions. The specified matrix will not be used")
    }
  }

  # reset L0 if necessary
  if (L0 > ncol(w)) {
    L0 <- ncol(w)
    warning("L0 parameter was set to a value larger than the rank of the feature model. L0 has been set to the rank of the feature model and no truncation will be applied.")
  }
  if (L0 == 0 || is.null(L0) || is.na(L0)) L0 <- ncol(w)

  # check validity of range, coerce to a vector.
  # odd indices should give lower ranges, even indices should give upper ranges
  if (!is.list(range)) {
    if (length(range) == 0 || length(range) > 2) stop("if an array is specified for range, it must be of length 1 or 2 corresponding to c(min, max)")
    range <- sapply(range, as.double)
  } else {
    for (i in 1:length(range)) {
      if (length(range[[i]]) > 2 || length(range[[i]]) < 1) stop("Specified list of ranges had vector items not of length 1 or 2.")
      if (length(range[[i]]) == 1) rep(range[[i]], 2)
    }
    range <- sapply(unlist(range), as.double)
  }

  # check if A is a matrix
  if ("matrix" %in% class(A)) {
    # check if A is more than 90% sparse. If so, we'll coerce it to a dgCMatrix as this is faster in the projection step.
    sparsity <- sum(A == 0) / prod(dim(A))
    if (sparsity > 0.9) {
      A <- as(A, "dgCMatrix")
      if (!is.null(weights)) weights <- as.matrix(weights)
    } else {
      A <- as.matrix(A)
      if (!is.null(weights)) weights <- as.matrix(weights)
    }
  } else {
    # if A is not of class "matrix", check if it is also not of class "dgCMatrix"
    if (!("dgCMatrix" %in% class(A))) {
      # see if this class can be coerced to a dense or sparse matrix
      if (canCoerce(A, "dgCMatrix")) {
        A <- as(A, "dgCMatrix")
      } else if (canCoerce(A, "matrix")) {
        A <- as.matrix(A)
      } else stop("A was neither of nor coercible to class \"matrix\" or \"dgCMatrix\".")
    } else {
      # if A is a dgCMatrix, coerce weights to a dgCMatrix as well, if applicable
      if (!is.null(weights)) weights <- as(weights, "dgCMatrix")
    }
  }

  # check weights for proper dimensions
  if (!is.null(weights))
    if (dim(weights) != dim(A)) {
      weights <- NULL
      warning("weights was not of equal dimensions as A, weights are being set to NULL and tlearn will proceed without weighting")
    }

  # initialize h with the mean of non-zero values in A / mean(w)
  if (is.null(h.init)) {
    mean_A <- ifelse(class(A) == "dgCMatrix", mean(A@x), mean(A[A > 0]))
    h.init <- matrix(mean_A, nrow = ncol(w), ncol = ncol(A))
  } else h.init <- as.matrix(h.init)

  if (tol > 1e-5) warning("tolerance is set very high. Consider using a smaller value to ensure robust results")
  if (maxit < 10) warning("maximum number of permitted iterations is set very low. Consider using a larger value to ensure robust results")

  # solve for h and return the value (using the Rcpp functions in "src/solve.cpp")
  if (is.null(weights)) {
    if (class(A) == "matrix") {
      h <- Rcpp_Solve(h.init, t(w), A, ifelse(parallel, 0L, 1L), as.integer(L0), as.double(L1), as.double(L2), as.double(PE), sapply(values, as.double), as.integer(maxit), as.double(tol), as.double(min), as.double(max))
    } else if (class(A) == "dgCMatrix") {
      h <- Rcpp_SpSolve(h.init, t(w), A, ifelse(parallel, 0L, 1L), as.integer(L0), as.double(L1), as.double(L2), as.double(PE), sapply(values, as.double), as.integer(maxit), as.double(tol), as.double(min), as.double(max))
    }
  } else {
    if (class(A) == "matrix") {
      h <- Rcpp_WSolve(h.init, t(w), A, weights, ifelse(parallel, 0L, 1L), as.integer(L0), as.double(L1), as.double(L2), as.double(PE), sapply(values, as.double), as.integer(maxit), as.double(tol), as.double(min), as.double(max))
    } else if (class(A) == "dgCMatrix") {
      h <- Rcpp_SpWSolve(h.init, t(w), A, weights, ifelse(parallel, 0L, 1L), as.integer(L0), as.double(L1), as.double(L2), as.double(PE), sapply(values, as.double), as.integer(maxit), as.double(tol), as.double(min), as.double(max))
    }
  }

  if(!is.null(colnames(A))) colnames(h) <- colnames(A)
  if(!is.null(colnames(w))) rownames(h) <- colnames(w)

  return(h)
}