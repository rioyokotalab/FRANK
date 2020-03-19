#include "hicma/operations/misc/get_dim.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/low_rank_shared.h"
#include "hicma/classes/hierarchical.h"

#include "yorel/yomm2/cute.hpp"

#include <iostream>


namespace hicma
{

int get_n_rows(const Node& A) {
  return get_n_rows_omm(A);
}

define_method(int, get_n_rows_omm, (const Dense& A)) {
  return A.dim[0];
}

define_method(int, get_n_rows_omm, (const LowRank& A)) {
  return A.dim[0];
}

define_method(int, get_n_rows_omm, (const LowRankShared& A)) {
  return A.dim[0];
}

define_method(int, get_n_rows_omm, (const Hierarchical& A)) {
  int n_rows = 0;
  for (int i=0; i<A.dim[0]; i++) {
    n_rows += get_n_rows(A(i, 0));
  }
  return n_rows;
}

define_method(int, get_n_rows_omm, (const Node& A)) {
  std::cerr << "get_n_rows(";
  std::cerr << A.type();
  std::cerr << ") undefined." << std::endl;
  abort();
}


int get_n_cols(const Node& A) {
  return get_n_cols_omm(A);
}

define_method(int, get_n_cols_omm, (const Dense& A)) {
  return A.dim[1];
}

define_method(int, get_n_cols_omm, (const LowRank& A)) {
  return A.dim[1];
}

define_method(int, get_n_cols_omm, (const LowRankShared& A)) {
  return A.dim[1];
}

define_method(int, get_n_cols_omm, (const Hierarchical& A)) {
  int n_cols = 0;
  for (int j=0; j<A.dim[1]; j++) {
    n_cols += get_n_cols(A(0, j));
  }
  return n_cols;
}

define_method(int, get_n_cols_omm, (const Node& A)) {
  std::cerr << "get_n_cols(";
  std::cerr << A.type();
  std::cerr << ") undefined." << std::endl;
  abort();
}

} // namespace hicma
