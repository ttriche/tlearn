#' Cosine similarity of samples to a feature model
#'
#' Cosine similarity of samples to each factor in a feature model can be used as a measure of purity of feature expression in each sample, or for matching. It is not a dimensional reduction.
#'
#' Euclidean-norm cosine distance is much faster than Pearson correlation on sparse matrices (2-3x faster) due to fast vector operations and gives very similar (but unsigned) results.
#' For non-negative data, cosine similarity is ideal because it is constrained to the positive orthant (with exceptions for random matrices in the case of Euclidean-norm cosine similarity).
#'
#' Distance-based measures provide different information than a least squares-based projection: 1) the distances are measures of affinity and not additivity and thus 2) a distance measure gives the purity of a factor in a sample, not the representation of that factor in a sample.
#'
#' @param A Samples matrix of features x samples, may be dense or sparse
#' @param w feature model matrix of features x factors, may be dense or sparse
#' @returns matrix of same dimensions as h, giving the cosine distance between all factors and all samples
#' @export
#'
cosine <- function(A, w) {
  return(as.matrix(t(
    crossprod(
        tcrossprod(
            A,
            Diagonal(x = as.vector(crossprod(A ^ 2, rep(1, nrow(A)))) ^ -0.5)
        ),
        tcrossprod(
            w,
            Diagonal(x = as.vector(crossprod(w ^ 2, rep(1, nrow(w)))) ^ -0.5)
        )
    )
  )))
}