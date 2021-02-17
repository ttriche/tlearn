#include "tlearn.h"

//' @name Rcpp_bls
//' @title Bounded least squares
//' @description Solve \code{a \%*\% x = b} using sequential coordinate descent bounded least squares, where \code{a} is a square matrix and \code{x} and \code{b} are vectors.
//' @details See \code{cls} R function documentation for a more convenient interface and details. This function should only be used when speed is paramount. Fields are not nullable.
//' @param x required initial numeric vector of doubles for x, may simply specify a vector of ones
//' @param a square numeric dense matrix of doubles containing the coefficients of the linear system
//' @param b numeric vector of doubles giving the right-hand side of the linear system
//' @param maxit integer, maximum number of iterations for least squares sequential coordinate descent
//' @param tol double, tolerance for convergence of the least squares sequential coordinate descent solver
//' @param min double minimum permitted value, use zero for NMF
//' @param max double, maximum permitted value, use a large value to avoid an upper bound constraint
//' @param fixed a vector of 0 or 1 corresponding to "x" indicating if values in the supplied "x" vector should be fixed (i.e. not updated). To fix, specify 1, otherwise specify 0. If not fixing any values, specify a vector of zeros of length x.
//' @return x vector of least squares solution
//' @export
//[[Rcpp::export]]
arma::vec Rcpp_bls(arma::vec x, const arma::mat& a, arma::vec& b, const unsigned int maxit, const double tol, const double min, const double max, const arma::vec fixed) {
    double xi;
    double exi = 0;
    double er = 1;
    for (unsigned int t = 0; t < maxit && er > tol; t++) {
        er = 0;
        // loop through all columns of B and all values in vector a to find best values for x
        for (unsigned int k = 0; k < a.n_cols; k++) {
            if (fixed(k) == 0) {
                // calculate new value for index k of vector x
                xi = x(k) - b(k) / a(k, k);
                // constrain x to within the bounds of [min, max]
                if (xi < min) xi = min;
                else if (xi > max) xi = max;
                if (xi != x(k)) {
                    // update error and descent coordinates
                    b += (xi - x(k)) * a.col(k);
                    exi = 2 * std::abs(x(k) - xi) / (xi + x(k) + 1e-16);
                    if (exi > er) er = exi;
                    // update value in x
                    x(k) = xi;
                }
            }
        }
    }
    return (x);
}

// return closest number in "values" to x, "values" is ascending and of at least length 2.  Helper function for multinomial_bls.
double round_to_values(const double x, const arma::vec values) {
    double r = values(0);
    double r_err = std::abs(x - values(0));
    double i_err;
    // ascending search loop, breaks after passing the value of x
    for (unsigned int i = 1; i < values.n_elem; i++) {
        i_err = std::abs(x - values(i));
        if (i_err < r_err) {
            r_err = i_err;
            r = values(i);
        }
        if (values(i) > x) break;
    }
    return(r);
}

//' @name Rcpp_multinomial_bls
//' @title Multinomial-constrained bounded least squares
//' @description Solve \code{a \%*\% x = b} using sequential coordinate descent bounded least squares, where \code{a} is a square matrix and \code{x} and \code{b} are vectors and \code{x} is constrained to values in a specified multinomial distribution
//' @details See \code{cls} R function documentation for a more convenient interface and details. This function should only be used when speed is paramount. Fields are not nullable. \code{multinomial_bls} first fits a simple bounded model, then takes the largest value and fixes it to the closest value in the multinomial distribution. This process is repeated until until all values have been constrained.
//' @param x required initial numeric vector of doubles for x, may simply specify a vector of ones
//' @param a square numeric dense matrix of doubles containing the coefficients of the linear system
//' @param b numeric vector of doubles giving the right-hand side of the linear system
//' @param maxit integer, maximum number of iterations for least squares sequential coordinate descent
//' @param tol double, tolerance for convergence of the least squares sequential coordinate descent solver
//' @param min double minimum permitted value, use zero for NMF
//' @param max double, maximum permitted value, use a large value to avoid an upper bound constraint
//' @param fixed a vector of 0 or 1 corresponding to "x" indicating if values in the supplied "x" vector should be fixed (i.e. not updated). To fix, specify 1, otherwise specify 0. If not fixing any values, specify a vector of zeros of length x.
//' @param values numeric array, either a vector of c(0, 0) to ignore multinomial constraints, or a vector of permitted values in the projection. Must be specified in ascending order and be of at least length 2 and strictly in the range [min, max].
//' @return x vector of least squares solution
//' @export
//[[Rcpp::export]]
arma::vec Rcpp_multinomial_bls(arma::vec x, const arma::mat& a, arma::vec& b, const unsigned int maxit, const double tol, const double min, const double max, const arma::vec values, arma::vec fixed) {

    // first solve the bounded equation
    x = Rcpp_bls(x, a, b, maxit, tol, min, max, fixed);

    // next set the largest value to a value from the multinomial distribution and fix it. Repeat for all values in x
    for (unsigned int i = 0; i < x.n_elem; i++) {

        // get index of maximum unfixed value in x
        unsigned int max_index;
        double ind_max = 0;
        for (unsigned int j = 0; j < x.n_elem; j++) {
            if (fixed(j) == 0 && x(j) > ind_max) {
                ind_max = x(j);
                max_index = j;
            }
        }

        // if maximum unfixed value in x is equal to minimum value in values distribution (usually zero), we're done restting values. Fix that index.
        if (x(max_index) == values.min()) break;
        fixed(max_index) = 1;
        x(max_index) = round_to_values(x(max_index), values);

        // refit the model with the fixed value
        x = Rcpp_bls(x, a, b, maxit, tol, min, max, fixed);
    }
    return(x);
}

//' @name Rcpp_L0_bls
//' @title L0 and/or multinomial-constrained bounded least squares
//' @description Solve \code{a \%*\% x = b} using sequential coordinate descent bounded least squares, where \code{a} is a square matrix and \code{x} and \code{b} are vectors and \code{x} is constrained to values in a specified multinomial distribution and the cardinality of x is truncated to specified L0 value.
//' @details See \code{cls} R function documentation for a more convenient interface and details. This function should only be used when speed is paramount. Fields are not nullable. \code{multinomial_bls} first fits a simple bounded model, then takes the largest value and fixes it to the closest value in the multinomial distribution. This process is repeated until until all values have been constrained. L0 regularization is applied to the \code{multinomial_bls} fit by incrementally imposing the penalty and refitting the model with each step into the full truncation.
//' @param x required initial numeric vector of doubles for x, may simply specify a vector of ones
//' @param a square numeric dense matrix of doubles containing the coefficients of the linear system
//' @param b numeric vector of doubles giving the right-hand side of the linear system
//' @param maxit integer, maximum number of iterations for least squares sequential coordinate descent
//' @param tol double, tolerance for convergence of the least squares sequential coordinate descent solver
//' @param min double minimum permitted value, use zero for NMF
//' @param max double, maximum permitted value, use a large value to avoid an upper bound constraint
//' @param fixed a vector of 0 or 1 corresponding to "x" indicating if values in the supplied "x" vector should be fixed (i.e. not updated). To fix, specify 1, otherwise specify 0. If not fixing any values, specify a vector of zeros of length x.
//' @param values numeric array, either a vector of c(0, 0) to ignore multinomial constraints, or a vector of permitted values in the projection. Must be specified in ascending order and be of at least length 2 and strictly in the range [min, max].
//' @param L0 the cardinality or L0 truncation to impose on x. This will be incrementally enforced.
//' @return x vector of least squares solution
//' @export
//[[Rcpp::export]]
arma::vec Rcpp_L0_bls(arma::vec x, const arma::mat& a, arma::vec& b, const unsigned int maxit, const double tol, const double min, const double max, const arma::vec values, const double L0, arma::vec fixed) {

    // start by fitting the untruncated model
    x = (sum(values) == 0) ? Rcpp_bls(x, a, b, maxit, tol, min, max, fixed) : Rcpp_multinomial_bls(x, a, b, maxit, tol, min, max, values, fixed);

    // get the number of non-zero values in x
    arma::vec x_nz = nonzeros(x);

    // incrementally set values to zero, fix them, fit the model, repeat until the necessary number of values have been set to zero
    while (x_nz.n_elem > L0) {
        // get index of minimum non-zero value in x, set to zero, and fix it
        uvec min_index = find(x == x_nz.min()); // index of minimum non-zero value in x;
        x(min_index(0)) = 0;
        fixed(min_index(0)) = 1;

        // fit the model and update number of non-zero values in the model
        x = (sum(values) == 0) ? Rcpp_bls(x, a, b, maxit, tol, min, max, fixed) : Rcpp_multinomial_bls(x, a, b, maxit, tol, min, max, values, fixed);
        x_nz = nonzeros(x);
    }
    return(x);
}

//[[Rcpp::export]]
arma::vec Rcpp_cls(arma::vec x, const arma::mat& a, arma::vec& b, const unsigned int maxit, const double tol, const double min, const double max, const arma::vec values, const double L0) {

    // create a vector of "fixed" values in x, where fixed is designated by a 1
    arma::vec fixed = zeros<vec>(x.n_elem);

    // this is at least a L0 regularization truncation. The L0 function handles multinomial constraints
    if (L0 != 0 && L0 < x.n_elem)
        x = Rcpp_L0_bls(x, a, b, maxit, tol, min, max, values, L0, fixed);

    // this is not an L0 regularization truncation but is a multinomial distribution constraint
    else if (sum(values) > 0)
        x = Rcpp_multinomial_bls(x, a, b, maxit, tol, min, max, values, fixed);

    // this is purely a bounded least squares problem
    else if (sum(values) == 0)
        x = Rcpp_bls(x, a, b, maxit, tol, min, max, fixed);

    return(x);
}