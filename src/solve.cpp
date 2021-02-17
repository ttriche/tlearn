#include "tlearn.h"

//' @name Rcpp_Solve
//' @title Solve A = WH for H
//' @description See "tlearn" R function documentation for practical details. This function requires full specification as described below. Fields are not nullable.
//' @param h initial sample embeddings model (as factors x samples)
//' @param wt transposed feature model dense matrix (as factors x features, note transposition)
//' @param A features x samples matrix in dense format
//' @param threads integer, threads number of threads to use, or 0 to use all threads as determined by OpenMP
//' @param L0 integer, L0 sample truncation
//' @param L1 double, L1/Lasso regularization
//' @param L2 double, L2/ridge regression
//' @param PE double, pattern extraction regularization
//' @param values numeric array, either a vector of c(0, 0) to ignore multinomial constraints, or a vector of permitted values in the projection
//' @param maxit integer, number of iterations permitted for the sequential coordinate descent least squares solver
//' @param tol double, tolerance for convergence of sequential coordinate descent least squares solver
//' @param min double minimum permitted value in model (usually 0)
//' @param max double maximum permitted value in model (set to very large value to avoid an upper bound)
//' @return h matrix of sample embeddings
//' @export
//[[Rcpp::export]]
arma::mat Rcpp_Solve(arma::mat& h, const arma::mat& wt, const arma::mat& A, const int threads, const int L0, const double L1, const double L2, const double PE, const arma::vec values, const int maxit, const double tol, const double min, const double max) {
    arma::mat wtw = wt * wt.t();
    wtw.diag() += L2 != PE ? L2 - PE + 1e-16 : 1e-16;
    if (PE != 0) wtw += PE;

    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int j = 0; j < A.n_cols; j++) {
        arma::vec mu = wtw * h.col(j) - (wt * arma::vec(A.col(j))) + L1;
        h.col(j) = Rcpp_cls(h.col(j), wtw, mu, maxit, tol, min, max, values, L0);
    }
    return(h);
}

//' @name Rcpp_SpSolve
//' @title Solve <SpMat>A = WH for H
//' @description See "tlearn" R function documentation for practical details. This function requires full specification as described below. Fields are not nullable.
//' @param h initial sample embeddings model (as factors x samples)
//' @param wt transposed feature model dense matrix (as factors x features, note transposition)
//' @param A sparse matrix of features x samples
//' @param threads integer, threads number of threads to use, or 0 to use all threads as determined by OpenMP
//' @param L0 integer, L0 sample truncation
//' @param L1 double, L1/Lasso regularization
//' @param L2 double, L2/ridge regression
//' @param PE double, pattern extraction regularization
//' @param values numeric array, either a vector of c(0, 0) to ignore multinomial constraints, or a vector of permitted values in the projection
//' @param maxit integer, number of iterations permitted for the sequential coordinate descent least squares solver
//' @param tol double, tolerance for convergence of sequential coordinate descent least squares solver
//' @param min double minimum permitted value in model (usually 0)
//' @param max double maximum permitted value in model (set to very large value to avoid an upper bound)
//' @return h matrix of sample embeddings
//' @export
//' @export
//[[Rcpp::export]]
arma::mat Rcpp_SpSolve(arma::mat& h, const arma::mat& wt, const arma::SpMat<double>& A, const int threads, const int L0, const double L1, const double L2, const double PE, const arma::vec values, const int maxit, const double tol, const double min, const double max) {
    arma::mat wtw = wt * wt.t();
    wtw.diag() += L2 != PE ? L2 - PE + 1e-16 : 1e-16;
    if (PE != 0) wtw += PE;

    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int j = 0; j < A.n_cols; j++) {
        arma::vec mu = wtw * h.col(j) - (wt * arma::vec(A.col(j))) + L1;
        h.col(j) = Rcpp_cls(h.col(j), wtw, mu, maxit, tol, min, max, values, L0);
    }
    return(h);
}

//' @name Rcpp_WSolve
//' @title Solve A = WH for H with weighting
//' @description See "tlearn" R function documentation for practical details. This function requires full specification as described below. Fields are not nullable.
//' @param h initial sample embeddings model (as factors x samples)
//' @param wt transposed feature model dense matrix (as factors x features, note transposition)
//' @param A dense matrix of features x samples
//' @param A_weights dense matrix of weights for features x samples
//' @param threads integer, threads number of threads to use, or 0 to use all threads as determined by OpenMP
//' @param L0 integer, L0 sample truncation
//' @param L1 double, L1/Lasso regularization
//' @param L2 double, L2/ridge regression
//' @param PE double, pattern extraction regularization
//' @param values numeric array, either a vector of c(0, 0) to ignore multinomial constraints, or a vector of permitted values in the projection
//' @param maxit integer, number of iterations permitted for the sequential coordinate descent least squares solver
//' @param tol double, tolerance for convergence of sequential coordinate descent least squares solver
//' @param min double minimum permitted value in model (usually 0)
//' @param max double maximum permitted value in model (set to very large value to avoid an upper bound)
//' @return h matrix of sample embeddings
//' @export
//[[Rcpp::export]]
arma::mat Rcpp_WSolve(arma::mat& h, const arma::mat& wt, const arma::mat& A, const arma::mat& A_weights, const int threads, const int L0, const double L1, const double L2, const double PE, const arma::vec values, const int maxit, const double tol, const double min, const double max) {
    // apply weights to w
    arma::mat A_weighted = A_weights * A;

    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int j = 0; j < A.n_cols; j++) {
        arma::vec Aj = arma::vec(A_weighted.col(j));
        arma::vec Ajw = arma::vec(A_weights.col(j));
        arma::uvec nz = find(Ajw > 0);
        arma::mat wtj = wt.cols(nz);
        Aj = Aj.elem(nz);
        Ajw = Ajw.elem(nz);
        Aj = Aj.elem(nz);
        // if non-zero weights are not exclusively non-one, multiply columns in wt by A_weights
        if (Ajw.min() != 1) for (unsigned int ind = 0; ind < wtj.n_cols; ind++) wtj.col(ind) *= Ajw(ind);
        arma::mat wtw = wtj * wtj.t();
        wtw.diag() += L2 != PE ? L2 - PE + 1e-16 : 1e-16;
        if (PE != 0) wtw += PE;
        arma::vec mu = (wtw * h.col(j) - wtj * Aj) + L1;
        h.col(j) = Rcpp_cls(h.col(j), wtw, mu, maxit, tol, min, max, values, L0);
    }
    return(h);
}

//' @name Rcpp_SpWSolve
//' @title Solve <SpMat>A = WH for H with weighting
//' @description See "tlearn" R function documentation for practical details. This function requires full specification as described below. Fields are not nullable.
//' @param h initial sample embeddings model (as factors x samples)
//' @param wt transposed feature model dense matrix (as factors x features, note transposition)
//' @param A sparse matrix of features x samples
//' @param A_weights sparse matrix of weights for features x samples
//' @param threads integer, threads number of threads to use, or 0 to use all threads as determined by OpenMP
//' @param L0 integer, L0 sample truncation
//' @param L1 double, L1/Lasso regularization
//' @param L2 double, L2/ridge regression
//' @param PE double, pattern extraction regularization
//' @param values numeric array, either a vector of c(0, 0) to ignore multinomial constraints, or a vector of permitted values in the projection
//' @param maxit integer, number of iterations permitted for the sequential coordinate descent least squares solver
//' @param tol double, tolerance for convergence of sequential coordinate descent least squares solver
//' @param min double minimum permitted value in model (usually 0)
//' @param max double maximum permitted value in model (set to very large value to avoid an upper bound)
//' @return h matrix of sample embeddings
//' @export
//[[Rcpp::export]]
arma::mat Rcpp_SpWSolve(arma::mat& h, const arma::mat& wt, const arma::SpMat<double>& A, const arma::SpMat<double>& A_weights, const int threads, const int L0, const double L1, const double L2, const double PE, const arma::vec values, const int maxit, const double tol, const double min, const double max) {
    // apply weights to w
    arma::SpMat<double> A_weighted = A_weights;
    sp_mat::const_iterator it = A_weighted.begin();
    sp_mat::const_iterator it_end = A_weighted.end();
    for (; it != it_end; ++it) A_weighted.at(it.row(), it.col()) *= A.at(it.row(), it.col());

    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int j = 0; j < A.n_cols; j++) {
        arma::vec Aj = arma::vec(A_weighted.col(j));
        arma::vec Ajw = arma::vec(A_weights.col(j));
        arma::uvec nz = find(Ajw > 0);
        arma::mat wtj = wt.cols(nz);
        Aj = Aj.elem(nz);
        Ajw = Ajw.elem(nz);
        Aj = Aj.elem(nz);
        // if non-zero weights are not exclusively non-one, multiply columns in wt by A_weights
        if (Ajw.min() != 1) for (unsigned int ind = 0; ind < wtj.n_cols; ind++) wtj.col(ind) *= Ajw(ind);
        arma::mat wtw = wtj * wtj.t();
        wtw.diag() += L2 != PE ? L2 - PE + 1e-16 : 1e-16;
        if (PE != 0) wtw += PE;
        arma::vec mu = (wtw * h.col(j) - wtj * Aj) + L1;
        h.col(j) = Rcpp_cls(h.col(j), wtw, mu, maxit, tol, min, max, values, L0);
    }
    return(h);
}