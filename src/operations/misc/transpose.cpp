#include "hicma/operations/misc/transpose.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"

#include <utility>


namespace hicma
{

void transpose(Node& A) {
  transpose_omm(A);
}

define_method(void, transpose_omm, (Dense& A)) {
  // This implementation depends heavily on the details of Dense,
  // thus handled inside the class.
  A.transpose();
}

define_method(void, transpose_omm, (LowRank& A)) {
  using std::swap;
  transpose(A.U());
  transpose(A.S());
  transpose(A.V());
  swap(A.dim[0], A.dim[1]);
  swap(A.U(), A.V());
}

define_method(void, transpose_omm, (Hierarchical& A)) {
  using std::swap;
  Hierarchical A_trans(A.dim[1], A.dim[0]);
  for(int i=0; i<A.dim[0]; i++) {
    for(int j=0; j<A.dim[1]; j++) {
      swap(A(i, j), A_trans(j, i));
      transpose(A_trans(j, i));
    }
  }
  swap(A, A_trans);
}

define_method(void, transpose_omm, (Node& A)) {
  omm_error_handler("transpose", {A}, __FILE__, __LINE__);
  abort();
}

} // namespace hicma
