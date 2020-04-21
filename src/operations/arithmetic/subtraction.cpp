#include "hicma/operations/arithmetic.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/operations/misc/get_dim.h"

#include <cassert>
#include <cstdint>


namespace hicma
{

NodeProxy operator-(const Node& A, const Node& B) {
  assert(get_n_rows(A) == get_n_rows(B));
  assert(get_n_cols(A) == get_n_cols(B));
  return subtraction_omm(A, B);
}

define_method(NodeProxy, subtraction_omm, (const Dense& A, const Dense& B)) {
  Dense out(A.dim[0], A.dim[1]);
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      out(i, j) = A(i, j) - B(i, j);
    }
  }
  return out;
}

} // namespace hicma
