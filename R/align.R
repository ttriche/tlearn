#' Align two factor models
#'
#' Match factors between two matrices of "features x factors" on a cosine adjacency matrix with bipartite graph matching
#'
#' Given two factor models "w" and "wref", factors (columns) in "w" are reordered to correspond to the best possible matching with factors in "wref". 
#'
#' Generally both models will be of the same rank. If they are not of the same rank, the matching will be underdetermined and "w" will be truncated to a rank of the lesser of ncol(w) or ncol(wref).
#'
#' Models are aligned along the short edge, thus factors must be less than features/samples.
#'
#' A cosine adjacency matrix is used as the cost matrix for bipartite graph matching using the Hungarian algorithm. For details on the cosine matrix construction, see \code{tlearn::purity}. For details on the bipartite graph matching algorithm, see the \code{RcppHungarian} R package.
#'
#' Attention should be paid to the cosine distances between matched factors. Factors with low cosine similarity may still be matched because the matching minimizes the overall cost function, but have no affiliation with one another.
#'
#' Bipartite matching does not account for hierarchical relationships between factors. Thus, factors should generally be learned from datasets of comparable complexity and represent models of similar rank.
#'
#' @param m factor model to align to reference
#' @param m_ref reference factor model
#' @returns list of components:
#' \itemize{
#' \item m: reordered m matrix
#' \item pairs: pairings between the reference matrix and the template matrix
#' \item cost: cost of the matching on a cosine adjacency matrix
#' \item costMatrix: The cosine distance matrix between m and m_ref
#' \item dist: the cosine distance between each pair of matched factors
#' }
#' 
#' @seealso \code{\link[RcppHungarian]{HungarianSolver}}
#' @export
#'
align <- function(m, m_ref) {
  if(ncol(m) < nrow(m)){
    costMatrix <- 1 - cosine(m, m_ref) + 1e-10
  } else {
    costMatrix <- 1 - cosine(t(m), t(m_ref)) + 1e-10
  }
  costMatrix[costMatrix < 0] <- 0
  res <- bipartiteMatch(costMatrix)
  res$costMatrix <- costMatrix[, res$pairs[,2]]
  if(ncol(m) < nrow(m)){
    res$m <- m[, res$pairs[, 2]]
  } else {
    res$m <- m[res$pairs[, 2], ]
  }
  res$dist <- diag(res$costMatrix)
  return(res)
}