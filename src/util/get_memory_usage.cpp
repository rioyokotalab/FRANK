#include "FRANK/util/get_memory_usage.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/classes/matrix_proxy.h"
#include "FRANK/util/omm_error_handler.h"
#include "FRANK/util/print.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cstdint>
#include <cstdlib>
#include <memory>


namespace FRANK
{

declare_method(
  unsigned long, get_memory_usage_omm, (virtual_<const Matrix&>, const bool)
)

unsigned long get_memory_usage(const Matrix& A, const bool include_structure) {
  const unsigned long memory_usage = get_memory_usage_omm(A, include_structure);
  return memory_usage;
}

define_method(
  unsigned long, get_memory_usage_omm,
  (const Dense& A, const bool include_structure)
) {
  unsigned long memory_usage = 0;
  memory_usage += A.dim[0]*A.dim[1]*sizeof(A[0]);
  if (include_structure) {
    memory_usage += sizeof(Dense);
  }
  return memory_usage;
}

define_method(
  unsigned long, get_memory_usage_omm,
  (const LowRank& A, const bool include_structure)
) {
  unsigned long memory_usage = 0;
  memory_usage += get_memory_usage_omm(A.U, include_structure);
  memory_usage += get_memory_usage_omm(A.S, include_structure);
  memory_usage += get_memory_usage_omm(A.V, include_structure);
  if (include_structure) {
    memory_usage += sizeof(LowRank) - sizeof(Dense);
  }
  return memory_usage;
}

define_method(
  unsigned long, get_memory_usage_omm,
  (const Hierarchical& A, const bool include_structure)
) {
  unsigned long memory_usage = 0;
  for (int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      memory_usage += get_memory_usage_omm(A(i, j), include_structure);
    }
  }
  if (include_structure) {
    memory_usage += A.dim[0] * A.dim[1] * sizeof(MatrixProxy);
    memory_usage += sizeof(Hierarchical);
  }
  return memory_usage;
}

define_method(unsigned long, get_memory_usage_omm, (const Matrix& A, const bool)) {
  omm_error_handler("get_memory_usage", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace FRANK
