#include "hicma/util/get_memory_usage.h"
#include "hicma/extension_headers/util.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/classes/intitialization_helpers/basis_tracker.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/print.h"

#include "yorel/yomm2/cute.hpp"

#include <cstdint>
#include <cstdlib>
#include <memory>


namespace hicma
{

unsigned long get_memory_usage(const Matrix& A, bool include_structure) {
  BasisTracker<BasisKey> tracker;
  return get_memory_usage(A, tracker, include_structure);
}

unsigned long get_memory_usage(
  const Matrix& A, BasisTracker<BasisKey>& tracker, bool include_structure
) {
  return get_memory_usage_omm(A, tracker, include_structure);
}

define_method(
  unsigned long, get_memory_usage_omm,
  (const Dense& A, BasisTracker<BasisKey>& tracker, bool include_structure)
) {
  // If already tracked, return size 0
  if (tracker.has_basis(A)) return 0;
  // Otherwise register with tracker and calculate size
  tracker[A] = MatrixProxy();
  unsigned long memory_usage = 0;
  memory_usage += A.dim[0]*A.dim[1]*sizeof(A[0]);
  if (include_structure) {
    memory_usage += sizeof(Dense);
  }
  return memory_usage;
}

define_method(
  unsigned long, get_memory_usage_omm,
  (
    const SharedBasis& A,
    BasisTracker<BasisKey>& tracker,
    bool include_structure
  )
) {
  // If transfer matrix is already registered, return 0
  if (tracker.has_basis(A.transfer_mat())) return 0;
  // Otherwise register with tracker, calculate size and recurse
  tracker[A.transfer_mat()] = MatrixProxy();
  unsigned long memory_usage = get_memory_usage(
    A.transfer_mat(), tracker, include_structure
  );
  for (int64_t i=0; i<A.num_child_basis(); ++i) {
    memory_usage += get_memory_usage(A[i], tracker, include_structure);
  }
  if (include_structure) {
    memory_usage += sizeof(SharedBasis);
  }
  return memory_usage;
}

define_method(
  unsigned long, get_memory_usage_omm,
  (const LowRank& A, BasisTracker<BasisKey>& tracker, bool include_structure)
) {
  unsigned long memory_usage = 0;
  memory_usage += get_memory_usage(A.U, tracker, include_structure);
  memory_usage += get_memory_usage(A.S, tracker, include_structure);
  memory_usage += get_memory_usage(A.V, tracker, include_structure);
  if (include_structure) {
    memory_usage += sizeof(LowRank) - 3*sizeof(Dense);
  }
  return memory_usage;
}

define_method(
  unsigned long, get_memory_usage_omm,
  (
    const Hierarchical& A,
    BasisTracker<BasisKey>& tracker,
    bool include_structure
  )
) {
  unsigned long memory_usage = 0;
  for (int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      memory_usage += get_memory_usage(A(i, j), tracker, include_structure);
    }
  }
  if (include_structure) {
    memory_usage += A.dim[0] * A.dim[1] * sizeof(MatrixProxy);
    memory_usage += sizeof(Hierarchical);
  }
  return memory_usage;
}

define_method(
  unsigned long, get_memory_usage_omm,
  (
    const Matrix& A,
    [[maybe_unused]] BasisTracker<BasisKey>& tracker,
    [[maybe_unused]] bool include_structure
  )
) {
  omm_error_handler("get_memor_usage", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
