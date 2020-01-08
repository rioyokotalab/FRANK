#include "hicma/util/get_memory_usage.h"

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/low_rank_shared.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/uniform_hierarchical.h"
#include "hicma/util/print.h"

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

#include <memory>

namespace hicma
{

unsigned long get_memory_usage(const Node& A, bool include_structure) {
  return get_memory_usage_omm(A, include_structure);
}

BEGIN_SPECIALIZATION(
  get_memory_usage_omm, unsigned long,
  const Dense& A, bool include_structure
) {
  unsigned long memory_usage = 0;
  memory_usage += A.dim[0]*A.dim[1]*sizeof(A[0]);
  if (include_structure) {
    memory_usage += sizeof(Dense);
  }
  return memory_usage;
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  get_memory_usage_omm, unsigned long,
  const LowRank& A, bool include_structure
) {
  unsigned long memory_usage = 0;
  memory_usage += get_memory_usage(A.U, include_structure);
  memory_usage += get_memory_usage(A.S, include_structure);
  memory_usage += get_memory_usage(A.V, include_structure);
  if (include_structure) {
    memory_usage += sizeof(LowRank) - 3*sizeof(Dense);
  }
  return memory_usage;
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  get_memory_usage_omm, unsigned long,
  const LowRankShared& A, bool include_structure
) {
  unsigned long memory_usage = 0;
  memory_usage += get_memory_usage(A.S, include_structure);
  if (include_structure) {
    memory_usage += sizeof(LowRankShared) - sizeof(Dense);
  }
  return memory_usage;
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  get_memory_usage_omm, unsigned long,
  const Hierarchical& A, bool include_structure
) {
  unsigned long memory_usage = 0;
  for (int i=0; i<A.dim[0]; ++i) {
    for (int j=0; j<A.dim[1]; ++j) {
      memory_usage += get_memory_usage(A(i, j), include_structure);
    }
  }
  if (include_structure) {
    memory_usage += A.dim[0] * A.dim[1] * sizeof(NodeProxy);
    memory_usage + sizeof(Hierarchical);
  }
  return memory_usage;
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  get_memory_usage_omm, unsigned long,
  const UniformHierarchical& A, bool include_structure
) {
  unsigned long memory_usage = 0;
  for (int i=0; i<A.dim[0]; ++i) {
    for (int j=0; j<A.dim[1]; ++j) {
      memory_usage += get_memory_usage(A(i, j), include_structure);
    }
  }
  for (int i=0; i<A.dim[0]; ++i) {
    memory_usage += get_memory_usage(A.get_row_basis(i), include_structure);
  }
  for (int j=0; j<A.dim[1]; ++j) {
    memory_usage += get_memory_usage(A.get_col_basis(j), include_structure);
  }
  // TODO consider moving these calculations to a class fuction!
  if (include_structure) {
    memory_usage += A.dim[0] * sizeof(std::shared_ptr<Dense>);
    memory_usage += A.dim[1] * sizeof(std::shared_ptr<Dense>);
    memory_usage += A.dim[0] * A.dim[1] * sizeof(NodeProxy);
    memory_usage += sizeof(UniformHierarchical);
  }
  return memory_usage;
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  get_memory_usage_omm, unsigned long,
  const Node& A, bool include_structure
) {
  std::cerr << "get_memory_usage(";
  std::cerr << A.type();
  std::cerr << ") undefined." << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma
