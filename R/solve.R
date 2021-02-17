#' Constrained least squares
#'
#' Generalizes non-negativity constraint to any range or set of range constraints, solves constrained least squares equations using fast sequential coordinate descent with parallelization in RcppArmadillo.
#'
#' Solve the equation \code{a \%*\% x = b} for \code{x} where \code{b} is a vector.
#'
#' If \code{b} is a matrix, \code{tlearn} will be called with default parameters. Use \code{tlearn} to take full advantage of additional regularizations, weights, etc.
#'
#' This function serves as an R interface to the least squares solver at the core of the \code{tlearn} transfer learning engine. The function is unique among available R/Python methods in it's ability to impose multiple ranges and implement fast least squares solving using sequential coordinate descent.
#'
#' The algorithm for sequential coordinate descent least squares was first introduced by Xihui Lin, R package \code{NNLM}. \code{tlearn} generalizes the non-negativity constraint in the original code to single- and multi-range constraints and improves speed.
#'
#' @param b numeric vector giving the right-hand side of the linear system
#' @param a square numeric matrix containing the coefficients of the linear system
#' @param x optional initial numeric vector for x
#' @param tol tolerance for convergence of the least squares sequential coordinate descent solver
#' @param maxit maximum number of iterations for least squares sequential coordinate descent
#' @param min minimum permitted value, zero for NMF
#' @param max maximum permitted value, default 1e10
#' @param values if applicable, a multinomial distribution of permitted values for "h" to be incrementally enforced, beginning with the largest value in a purely bounded solution for \code{x}
#' @param L0 L0 truncation to be incrementally enforced for each column in b, by default the length of x (ncol(b))
#' @returns vector of least squares solutions
#' @export
#'
cls <- function(a, b, x = NULL, maxit = 50, tol = 1e-8, min = 0, max = 1e10, values = NULL, L0 = ncol(b)) {
  # check validity of range, coerce to a vector.  Odd indices should give lower ranges, even indices should give upper ranges

  if(is.null(values)) values <- rep(0, 2)

  if (is.null(x)) x <- rep(mean(b) / mean(a), nrow(a))
  return(Rcpp_cls(as.vector(x), as.vector(b), as.matrix(a), as.integer(maxit), as.double(tol), as.double(min), as.double(max), sapply(values, as.double)), as.integer(L0))
}