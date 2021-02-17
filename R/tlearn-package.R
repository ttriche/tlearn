#' tlearn: Transfer Learn
#'
#' @description
#' Fast and flexible sample projection onto feature models for transfer learning with constraints, regularization, and weights
#'
#' @details 
#' Fast and flexible transfer learning.
#'
#' @section Why tlearn?:
#' \itemize{
#' \item fast parallelized constrained least squares coordinate descent solver
#' \item sparse or dense matrix support in dedicated backends
#' \item regularization (L1/Lasso, L2/Ridge, PE/Pattern Extraction) and truncation (L0 on the model or individual samples)
#' \item constraints to a single range, multiple ranges, or a multinomial distribution
#' \item weighting/masking
#' \item minibatching
#' \item utilities for post-projection analysis
#' }
#'
#' @section What is transfer learning?:
#' Given the right hand side of any matrix decomposition or feature model (matrix of features x factors), tlearn finds how represented each factor is in each sample. 
#' For example, a model describing facial features can be projected onto a set of face portraits to learn what features are present in that face. The resulting model will give a coefficient for each feature, and do so for each portrait.
#'
#' @section There's a vignette!:
#' The vignette succintly covers theoretical concepts, best practices, and demonstrates transfer learning using PCA, NMF, and UMAP as examples to explore single cell type inference and spatiotemporal patterns of bird species frequency.
#'
#' @section Functions in the tlearn package:
#' \itemize{
#' \item **tlearn**: project samples onto factor models
#' \item **tlearnMinibatch**: minibatch tlearn for large projections
#' \item **gnnls**: generalized constrained non-negative least squares, least squares solver behind tlearn
#' \item **loss**: error of a projection model
#' \item **rank**: rank features in a projection by total weight and scale "h" by the rank (diagonal) values
#' \item **sampleLoss**: error of a projection model for each sample. Find samples that are poorly or well explained by the factor model.
#' \item **align**: align factors in two models to facilitate direct comparisons
#' \item **purity**: cosine distance between samples and factors as a measure of factor expression purity in each sample
#' }
#'
#' @import knitr Matrix
#' @importFrom Rcpp evalCpp
#' @importFrom utils txtProgressBar setTxtProgressBar
#' @importFrom methods as canCoerce
#' @useDynLib tlearn, .registration = TRUE
#' @docType package
#' @name tlearn
#' @aliases tlearn-package
#' @md
#'
"_PACKAGE"