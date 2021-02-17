#define ARMA_NO_DEBUG

#ifdef _OPENMP
#include <omp.h>
#endif

//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

#include <Rcpp.h>
#include <R.h>

using namespace arma;

/******************************************************************************************************
 *
 * 2/15/2021 Zach DeBruine <zach.debruine@vai.org>
 * github.com/zdebruine/tlearn
 *
 * Please raise issues, feature requests, support/help requests on github.
 * Don't hesitate. I'll be happy to try and answer your questions!
 * 
 * ***************************************************************************************************/

 // bounded least squares solver
arma::vec Rcpp_bls(arma::vec x, const arma::mat& a, arma::vec& b, const unsigned int maxit, const double tol, const double min, const double max, const arma::vec fixed);

// multinomial bounded least squares solver
arma::vec Rcpp_multinomial_bls(arma::vec x, const arma::mat& a, arma::vec& b, const unsigned int maxit, const double tol, const double min, const double max, const arma::vec values, arma::vec fixed);

// L0 truncated multinomial bounded least squares solver
arma::vec Rcpp_L0_bls(arma::vec x, const arma::mat& a, arma::vec& b, const unsigned int maxit, const double tol, const double min, const double max, const arma::vec values, const double L0, arma::vec fixed);

// constrained least squares solver that shunts tasks to the appropriate solver above
arma::vec Rcpp_cls(arma::vec x, const arma::mat& a, arma::vec& b, const unsigned int maxit, const double tol, const double min, const double max, const arma::vec values, const double L0);

// solves an "A = wh" matrix factorization for "h" where "A" is dense
arma::mat Rcpp_Solve(arma::mat& h, const arma::mat& wt, const arma::mat& A, const int threads, const int L0, const double L1, const double L2, const double PE, const arma::vec values, const int maxit, const double tol, const double min, const double max);

// solves an "A = wh" matrix factorization for "h" where "A" is sparse
arma::mat Rcpp_SpSolve(arma::mat& h, const arma::mat& wt, const arma::SpMat<double>& A, const int threads, const int L0, const double L1, const double L2, const double PE, const arma::vec values, const int maxit, const double tol, const double min, const double max);

// solves an "A = wh" matrix factorization for "h" where "A" is dense and weighted
arma::mat Rcpp_WSolve(arma::mat& h, const arma::mat& wt, const arma::mat& A, const arma::mat& A_weights, const int threads, const int L0, const double L1, const double L2, const double PE, const arma::vec values, const int maxit, const double tol, const double min, const double max);

// solves an "A = wh" matrix factorization for "h" where "A" is sparse and weighted
arma::mat Rcpp_SpWSolve(arma::mat& h, const arma::mat& wt, const arma::SpMat<double>& A, const arma::SpMat<double>& A_weights, const int threads, const int L0, const double L1, const double L2, const double PE, const arma::vec values, const int maxit, const double tol, const double min, const double max);

// loss of "A - wh" factorization where "A" is dense and possibly weighted, adds regularization penalties
double Rcpp_Loss(const arma::mat w, const arma::vec d, const arma::mat h, const arma::mat& A, const arma::mat& A_weights, const double w_L1, const double w_L2, const double w_PE, const double h_L1, const double h_L2, const double h_PE, const unsigned int loss_type, const int threads);

// loss of "A - wh" factorization where "A" is sparse and possibly weighted, adds regularization penalties
double Rcpp_SpLoss(const arma::mat w, const arma::vec d, const arma::mat h, const arma::SpMat<double>& A, const arma::SpMat<double> A_weights, const double w_L1, const double w_L2, const double w_PE, const double h_L1, const double h_L2, const double h_PE, const unsigned int loss_type);

// loss of "A - wh" factorization for each sample where "A" is dense
arma::vec Rcpp_LossSample(const arma::mat w, const arma::vec d, const arma::mat h, const arma::mat& A, const unsigned int loss_type, const int threads);

// loss of "A - wh" factorization for each sample where "A" is sparse
arma::vec Rcpp_SpLossSample(const arma::mat w, const arma::vec d, const arma::mat h, const arma::SpMat<double>& A, const unsigned int loss_type);