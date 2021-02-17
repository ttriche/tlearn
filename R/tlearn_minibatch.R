#' Transfer learn in minibatches
#'
#' Given a very large sparse matrix of samples, minibatch tlearn reduces memory overhead by splitting the sample matrix into chunks and projecting one chunk at a time.
#'
#' You can achieve the same result by manually chunking the data and running tlearn on each chunk, then binding the chunks together again. Of course, model-wide regularization should be used with caution.
#'
#' Should you chunk?
#' * Pros: You get a progress bar, reduced memory overhead, and still just a single matrix as a result
#' * Cons: Takes fractions of a second longer, no support for weights, warm-start initialization, or model-wide regularization.
#'
#' @inheritParams tlearn
#' @param chunk_size number of samples to project at a time
#' @param verbose show progress bar
#' @param ... additional arguments to tlearn (use regularizations with caution)
#' @returns h, a sample embeddings matrix of dimensions "factors x samples".
#' @seealso \code{\link{tlearn}}
#' @export
#'
tlearn_minibatch <- function(A, w, chunk_size = 1000, verbose = TRUE, ...) {
  nchunks <- ceiling(ncol(A) / chunk_size)
  res <- list()
  if (verbose) pb <- txtProgressBar(min = 1, max = nchunks, style = 3)
  for (i in 1:nchunks) {
    min_ind <- i * chunk_size
    max_ind <- ifelse(min_ind + chunk_size > ncol(A), ncol(A), min_ind + chunk_size)
    res[[i]] <- tlearn(A[, c(min_ind:max_ind)], w, ...)
    if (verbose) setTxtProgressBar(pb, i)
  }
  return(do.call(cbind, res))
}