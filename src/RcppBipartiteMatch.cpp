#include <Rcpp.h>
#include "RcppHungarian.h"
#include <iostream>

using namespace Rcpp;

//' @name bipartiteMatch
//' @title Bipartite matching with the Hungarian algorithm 
//' @description Solves weighted bipartite matching on a cost matrix (i.e. distance) using the hungarian algorithm
//' @details This is a copy/clone of the code developed by Cong Ma and the Rcpp wrapper developed by Justin Silverman released in the RcppHungarian package.
//' @param costMatrix a cost matrix on which to match, note that the best matches will minimize the diagonal, thus distances are appropriate.
//' @return List with cost and pairings, pairings are given as an nx2 matrix giving edges that are metched.
//' @export
// [[Rcpp::export]]
List bipartiteMatch(NumericMatrix costMatrix) {
  int nr = costMatrix.nrow();
  int nc = costMatrix.ncol();
  
  vector<double> c(nc);
  vector<vector<double>> cm(nr, c);
  for (int i=0; i < nr; i++){
    for (int j=0; j < nc; j++){
      c[j] = costMatrix(i,j);
    }
    cm[i] = c;
  }
  
  HungarianAlgorithm HungAlgo;
  vector<int> assignment;
  double cost = HungAlgo.Solve(cm, assignment);
  IntegerMatrix assign(nr, 2);
  for (int i=0; i < nr; i++){
    assign(i,0) = i+1;
    assign(i,1) = assignment[i]+1;
  }
  List out(2);
  out[0] = cost;
  out[1] = assign;
  out.names() = CharacterVector::create("cost", "pairs");
  return out;
}

