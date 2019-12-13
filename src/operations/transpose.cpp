#include "hicma/operations/transpose.h"

#include "hicma/node.h"
#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"

#include "yorel/multi_methods.hpp"

namespace hicma
{

void transpose(Node& A) {
  transpose_omm(A);
}

BEGIN_SPECIALIZATION(transpose_omm, void, Dense& A) {
  // This implementation depends heavily on the details of Dense,
  // thus handled inside the class.
  A.transpose();
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(transpose_omm, void, LowRank& A) {
  using std::swap;
  transpose(A.U);
  transpose(A.S);
  transpose(A.V);
  swap(A.dim[0], A.dim[1]);
  swap(A.U, A.V);
} END_SPECIALIZATION;


BEGIN_SPECIALIZATION(transpose_omm, void, Hierarchical& A) {
  Hierarchical A_trans(A.dim[1], A.dim[0]);
  for(int i=0; i<A.dim[0]; i++) {
    for(int j=0; j<A.dim[1]; j++) {
      swap(A(i, j), A_trans(j, i));
      transpose(A_trans(j, i));
    }
  }
  swap(A, A_trans);
} END_SPECIALIZATION;


BEGIN_SPECIALIZATION(transpose_omm, void, Node& A) {
  std::cerr << "tranpose(";
  std::cerr << A.type();
  std::cerr << ") undefined." << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma
