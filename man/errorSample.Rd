% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/error.R
\name{errorSample}
\alias{errorSample}
\title{Model error or loss for each sample}
\usage{
errorSample(
  A,
  w,
  d = NULL,
  h,
  error_method = "mse",
  parallel = TRUE,
  relative = FALSE
)
}
\arguments{
\item{A}{samples to project: matrix of features x samples, dense or sparse}

\item{w}{feature coefficients model: matrix of features x factors}

\item{d}{model diagonal (i.e. scaling matrix in NMF, singular values in SVD/PCA), or simply NULL or a vector of ones if no diagonal is included in the model. Default is NULL.}

\item{h}{sample coefficients model: matrix of factors x samples}

\item{error_method}{"mse" or "mae" for mean squared/absolute error}

\item{parallel}{parallelize error loss calculation}

\item{relative}{return relative loss, the observed loss as a proportion of the squared sum (mse) or absolute sum (mae) of each sample}
}
\value{
numeric vector of loss for all samples
}
\description{
Mean squared or absolute error for each sample in a factorization
}
\details{
Useful to find how well samples fit (or do not fit) the model.
}
