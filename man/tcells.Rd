% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tcells.R
\docType{data}
\name{tcells}
\alias{tcells}
\title{Cluster centers for CD64+ T-cells}
\format{
matrix with features (genes) as rows and samples (cells) as columns
}
\source{
\href{https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133345}{GEO accession identifier}
}
\usage{
data(tcells)
}
\description{
This dataset summarizes gene expression signatures for 15 types of CD64+ T-cells, as described in Zhilei Bian et al. Nature 2020.
}
\details{
The dataset was obtained by downloading the T-cells data from Zhilei Bian, applying Seurat sctransform, and then averaging cells in each cluster.

It is provided as part of the \code{tlearn} package for projection onto other datasets.
}
\references{
Bian et al. Nature 2020
(\href{https://pubmed.ncbi.nlm.nih.gov/32499656/}{PubMed})
}
\keyword{datasets}
