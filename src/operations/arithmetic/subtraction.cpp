#include "hicma/operations/arithmetic.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>


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

define_method(NodeProxy, subtraction_omm, (const Node& A, const Node& B)) {
  omm_error_handler("operator-", {A, B}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
