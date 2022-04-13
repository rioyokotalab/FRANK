#include "hicma/util/get_memory_usage.h"
#include "hicma/extension_headers/util.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/print.h"

#include "yorel/yomm2/cute.hpp"

#include <cstdint>
#include <cstdlib>
#include <memory>


namespace hicma
{

unsigned long get_memory_usage(const Matrix& A, bool include_structure) {
  unsigned long memory_usage = get_memory_usage_omm(A, include_structure);
  return memory_usage;
}

template<typename T>
unsigned long get_memory_usage_dense(const Dense<T>& A, bool include_structure) {
  unsigned long memory_usage = 0;
  memory_usage += A.dim[0]*A.dim[1]*sizeof(A[0]);
  if (include_structure) {
    memory_usage += sizeof(Dense<T>);
  }
  return memory_usage;
}

define_method(
  unsigned long, get_memory_usage_omm,
  (const Dense<float>& A, bool include_structure)
) {
  return get_memory_usage_dense(A, include_structure);
}

define_method(
  unsigned long, get_memory_usage_omm,
  (const Dense<double>& A, bool include_structure)
) {
  return get_memory_usage_dense(A, include_structure);
}

template<typename T>
unsigned long get_memory_usage_low_rank(const LowRank<T>& A, bool include_structure) {
  unsigned long memory_usage = 0;
  memory_usage += get_memory_usage_omm(A.U, include_structure);
  memory_usage += get_memory_usage_omm(A.S, include_structure);
  memory_usage += get_memory_usage_omm(A.V, include_structure);
  if (include_structure) {
    memory_usage += sizeof(LowRank<T>) - sizeof(Dense<T>);
  }
  return memory_usage;
}

define_method(
  unsigned long, get_memory_usage_omm,
  (const LowRank<float>& A, bool include_structure)
) {
  return get_memory_usage_low_rank(A, include_structure);
}

define_method(
  unsigned long, get_memory_usage_omm,
  (const LowRank<double>& A, bool include_structure)
) {
  return get_memory_usage_low_rank(A, include_structure);
}

template<typename T>
unsigned long get_memory_usage_hierarchical(const Hierarchical<T>& A, bool include_structure) {
  unsigned long memory_usage = 0;
  for (int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      memory_usage += get_memory_usage_omm(A(i, j), include_structure);
    }
  }
  if (include_structure) {
    memory_usage += A.dim[0] * A.dim[1] * sizeof(MatrixProxy);
    memory_usage += sizeof(Hierarchical<T>);
  }
  return memory_usage;
}

define_method(
  unsigned long, get_memory_usage_omm,
  (const Hierarchical<float>& A, bool include_structure)
) {
  return get_memory_usage_hierarchical(A, include_structure);
}

define_method(
  unsigned long, get_memory_usage_omm,
  (const Hierarchical<double>& A, bool include_structure)
) {
  return get_memory_usage_hierarchical(A, include_structure);
}

define_method(unsigned long, get_memory_usage_omm, (const Matrix& A, bool)) {
  omm_error_handler("get_memory_usage", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
