#include "tlearn.h"

// helper function for c_error function family
double add_penalty(const arma::mat X, const double L1, const double L2, const double PE) {
    double loss = 0;
    if (L1 != 0) loss += L1 * sum(sum(X));
    if (L2 != PE) loss += 0.5 * (L2 - PE) * sum(sum(square(X)));
    if (PE != 0) loss += 0.5 * PE * sum(sum(X * X.t()));
    return(loss);
}

//' @name Rcpp_Loss
//' @title Loss of a factorization
//' @description Calculates the loss of "A - wdh" with optional weights and one-sided L1, L2, and PE regualrizations, parallelized across columns of "A", where "A" is dense.
//' @details This function is for cases where "A" is dense. Use the R function "error" for a more accessible interface and details. This is the Rcpp interface for the fastest possible implementation.
//' @param w matrix of features x factors (rows x columns)
//' @param d vector giving scaling diagonal between w and h, if no scaling specify a vector of 1s of the same length as the number of factors (i.e. number of columns in "w")
//' @param h matrix of factors x samples (rows x columns)
//' @param A dense matrix of features x samples (rows x columns)
//' @param A_weights weighting matrix for the loss, the function will not apply any weighting matrix that is not of the same dimensions as "A". For instance, if no weights are to be applied, simply specify a dense matrix of 1 with 1 row and column.
//' @param w_L1 L1/LASSO regularization on w
//' @param w_L2 L2/Ridge regression on w
//' @param w_PE PE/Pattern Extraction regularization on w
//' @param h_L1 L1/LASSO regularization on h
//' @param h_L2 L2/Ridge regression on h
//' @param h_PE PE/Pattern Extraction regularization on h
//' @param loss_type integer, 1 for "mse" or any other value for "mae"
//' @param threads number of threads for OpenMP parallelization, set to 0 to let OpenMP decide and use all available threads
//' @return loss of the factorization (as mse or mae)
//' @export
//[[Rcpp::export]]
double Rcpp_Loss(const arma::mat w, const arma::vec d, const arma::mat h, const arma::mat& A, const arma::mat& A_weights, const double w_L1, const double w_L2, const double w_PE, const double h_L1, const double h_L2, const double h_PE, const unsigned int loss_type, const int threads) {

    double loss = 0;

    // add penalties for regularizations
    if (w_L1 > 0 || w_L2 > 0 || w_PE > 0) loss += add_penalty(w, w_L1, w_L2, w_PE);
    if (h_L1 > 0 || h_L2 > 0 || h_PE > 0) loss += add_penalty(h, h_L1, h_L2, h_PE);

    // calculate square of difference between wdh and A for all non-zero values in A
    arma::mat wdh = (w * diagmat(d)) * h;

    if (threads == 0) {
#pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (unsigned int j = 0; j < A.n_cols; j++) {
            wdh.col(j) -= A.col(j);
        }
    } else {
        wdh -= A;
    }

    (loss_type == 1) ? wdh = square(wdh) : wdh = arma::abs(wdh);
    if (A_weights.n_cols == A.n_cols && A_weights.n_rows == A.n_rows) {
        // multiply residual wdh by the weights matrix
        loss += sum(sum(A_weights * wdh));
        return(loss / sum(sum(A_weights)));
    } else return((sum(sum(wdh)) + loss) / A.n_elem);
}

//' @name Rcpp_SpLoss
//' @title Loss of a factorization
//' @description Calculates the loss of "A - wdh" with optional weights and one-sided L1, L2, and PE regualrizations, parallelized across columns of "A", where "A" is sparse.
//' @details This function is for cases where "A" is a sparse matrix. Use the R function "error" for a more accessible interface and details. This is the Rcpp interface for the fastest possible implementation.
//' @param w matrix of features x factors (rows x columns)
//' @param d vector giving scaling diagonal between w and h, if no scaling specify a vector of 1s of the same length as the number of factors (i.e. number of columns in "w")
//' @param h matrix of factors x samples (rows x columns)
//' @param A sparse matrix of features x samples (rows x columns)
//' @param A_weights sparse weighting matrix for the loss, the function will not apply any weighting matrix that is not of the same dimensions as "A". For instance, if no weights are to be applied, simply specify a dense matrix of 1 with 1 row and column (of course, in sparse format).
//' @param w_L1 L1/LASSO regularization on w
//' @param w_L2 L2/Ridge regression on w
//' @param w_PE PE/Pattern Extraction regularization on w
//' @param h_L1 L1/LASSO regularization on h
//' @param h_L2 L2/Ridge regression on h
//' @param h_PE PE/Pattern Extraction regularization on h
//' @param loss_type integer, 1 for "mse" or any other value for "mae"
//' @param threads number of threads for OpenMP parallelization, set to 0 to let OpenMP decide and use all available threads
//' @return loss of the factorization (as mse or mae)
//' @export
//[[Rcpp::export]]
double Rcpp_SpLoss(const arma::mat w, const arma::vec d, const arma::mat h, const arma::SpMat<double>& A, const arma::SpMat<double> A_weights, const double w_L1, const double w_L2, const double w_PE, const double h_L1, const double h_L2, const double h_PE, const unsigned int loss_type) {

    double loss = 0;
    // add penalties for regularizations
    if (w_L1 > 0 || w_L2 > 0 || w_PE > 0) loss += add_penalty(w, w_L1, w_L2, w_PE);
    if (h_L1 > 0 || h_L2 > 0 || h_PE > 0) loss += add_penalty(h, h_L1, h_L2, h_PE);

    // calculate square of difference between wdh and A for all non-zero values in A
    arma::mat wdh = (w * diagmat(d)) * h;

    sp_mat::const_iterator it = A.begin();
    sp_mat::const_iterator it_end = A.end();
    for (; it != it_end; ++it) wdh.at(it.row(), it.col()) -= *it;
    (loss_type == 1) ? wdh = square(wdh) : wdh = arma::abs(wdh);
    if (A_weights.n_cols == A.n_cols && A_weights.n_rows == A.n_rows) {
        // multiply residual wdh by the weights matrix
        sp_mat::const_iterator itw = A_weights.begin();
        sp_mat::const_iterator itw_end = A_weights.end();
        for (; itw != itw_end; ++itw) loss += *itw * wdh.at(itw.row(), itw.col());
        return(loss / sum(sum(A_weights)));
    } else return((sum(sum(wdh)) + loss) / A.n_elem);
}

//' @name Rcpp_LossSample
//' @title Loss of a factorization for each sample
//' @description Calculates the loss of "A - wdh" for each sample, where "A" is dense.
//' @details This function is for cases where "A" is dense. Use the R function "errorSample" for a more accessible interface and details. This is the Rcpp interface for the fastest possible implementation.
//' @param w matrix of features x factors (rows x columns)
//' @param d vector giving scaling diagonal between w and h, if no scaling specify a vector of 1s of the same length as the number of factors (i.e. number of columns in "w")
//' @param h matrix of factors x samples (rows x columns)
//' @param A dense matrix of features x samples (rows x columns)
//' @param loss_type integer, 1 for "mse" or any other value for "mae"
//' @param threads number of threads for OpenMP parallelization, set to 0 to let OpenMP decide and use all available threads
//' @return loss of the factorization for each sample (as mse or mae)
//' @export
//[[Rcpp::export]]
arma::vec Rcpp_LossSample(const arma::mat w, const arma::vec d, const arma::mat h, const arma::mat& A, const unsigned int loss_type, const int threads) {
    arma::vec loss = vec(A.n_cols);
    arma::mat wdh = (w * diagmat(d)) * h;
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int j = 0; j < A.n_cols; j++) {
        if (loss_type == 1) {
            loss(j) = sum(square(A.col(j) - wdh.col(j)));
        } else {
            loss(j) = sum(arma::abs(A.col(j) - wdh.col(j)));
        }
    }
    return(loss/A.n_rows);
}


//' @name Rcpp_SpLossSample
//' @title Loss of a factorization for each sample
//' @description Calculates the loss of "A - wdh" for each sample, where "A" is sparse.
//' @details This function is for cases where "A" is sparse. Use the R function "errorSample" for a more accessible interface and details. This is the Rcpp interface for the fastest possible implementation.
//' @param w matrix of features x factors (rows x columns)
//' @param d vector giving scaling diagonal between w and h, if no scaling specify a vector of 1s of the same length as the number of factors (i.e. number of columns in "w")
//' @param h matrix of factors x samples (rows x columns)
//' @param A sparse matrix of features x samples (rows x columns)
//' @param loss_type integer, 1 for "mse" or any other value for "mae"
//' @param threads number of threads for OpenMP parallelization, set to 0 to let OpenMP decide and use all available threads
//' @return loss of the factorization for each sample (as mse or mae)
//' @export
//[[Rcpp::export]]
arma::vec Rcpp_SpLossSample(const arma::mat w, const arma::vec d, const arma::mat h, const arma::SpMat<double>& A, const unsigned int loss_type, const int threads) {
    arma::vec loss = vec(A.n_cols);
    arma::mat wdh = (w * diagmat(d)) * h;
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int j = 0; j < A.n_cols; j++) {
        if (loss_type == 1) {
            loss(j) = sum(square(arma::vec(A.col(j)) - wdh.col(j)));
        } else {
            loss(j) = sum(arma::abs(arma::vec(A.col(j)) - wdh.col(j)));
        }
    }
    return(loss/A.n_rows);
}