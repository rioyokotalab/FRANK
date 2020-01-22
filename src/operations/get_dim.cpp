#include "hicma/operations/get_dim.h"

#include "hicma/node.h"
#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"

#include "yorel/multi_methods.hpp"

namespace hicma
{

int get_n_rows(const Node& A) {
  return get_n_rows_omm(A);
}

BEGIN_SPECIALIZATION(get_n_rows_omm, int, const Dense& A) {
  return A.dim[0];
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(get_n_rows_omm, int, const LowRank& A) {
  return A.dim[0];
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(get_n_rows_omm, int, const Hierarchical& A) {
  int n_rows = 0;
  for (int i=0; i<A.dim[0]; i++) {
    n_rows += get_n_rows(A(i, 0));
  }
  return n_rows;
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(get_n_rows_omm, int, const Node& A) {
  std::cerr << "get_n_rows(";
  std::cerr << A.type();
  std::cerr << ") undefined." << std::endl;
  abort();
} END_SPECIALIZATION;


int get_n_cols(const Node& A) {
  return get_n_cols_omm(A);
}

BEGIN_SPECIALIZATION(get_n_cols_omm, int, const Dense& A) {
  return A.dim[1];
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(get_n_cols_omm, int, const LowRank& A) {
  return A.dim[1];
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(get_n_cols_omm, int, const Hierarchical& A) {
  int n_cols = 0;
  for (int j=0; j<A.dim[1]; j++) {
    n_cols += get_n_cols(A(0, j));
  }
  return n_cols;
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(get_n_cols_omm, int, const Node& A) {
  std::cerr << "get_n_cols(";
  std::cerr << A.type();
  std::cerr << ") undefined." << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma
