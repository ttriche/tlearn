#' Rank features by importance
#'
#' Finds a diagonal vector \code{d} such that rows in \code{h} sum to 1 when multiplied by \code{d}. In other words, the diagonal vector gives the relative weights of each factor across all samples in the model and a scaling of \code{h} which aids interpretability.
#'
#' The scaling is applied to the short edge of h. This assumes that the number of samples/features is greater than the rank.
#'
#' @param h sample factor model
#' @param d NULL, or the existing diagonal to update
#' @returns list of "d" and "h". "d" encodes the scaling diagonal vector (rank of each feature), and "h" is the newly scaled sample embeddings matrix.
#' @export
#'
rank <- function(h, d = NULL) {
  if (nrow(h) < ncol(h)) {
    if (is.null(d)) d <- rep(1, nrow(h))
    scaling <- 1 / rowSums(h)
    return(list("d" = d * scaling, "h" = h %*% Diagonal(x = scaling)))
  } else {
    if (is.null(d)) d <- rep(1, ncol(h))
    scaling <- 1 / colSums(h)
    return(list("d" = d * scaling, "h" = Diagonal(x = scaling) %*% h))
  }
}