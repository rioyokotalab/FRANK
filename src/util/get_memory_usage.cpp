#include "hicma/util/get_memory_usage.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/low_rank_shared.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/uniform_hierarchical.h"
#include "hicma/util/print.h"

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

unsigned long get_memory_usage(const Node& A) {
  return get_memory_usage_omm(A);
}

BEGIN_SPECIALIZATION(
  get_memory_usage_omm, unsigned long,
  const Dense& A
) {
  return A.dim[0]*A.dim[1]*sizeof(A[0]);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  get_memory_usage_omm, unsigned long,
  const LowRank& A
) {
  unsigned long memory_usage = 0;
  memory_usage += get_memory_usage(A.U);
  memory_usage += get_memory_usage(A.S);
  memory_usage += get_memory_usage(A.V);
  return memory_usage;
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  get_memory_usage_omm, unsigned long,
  const LowRankShared& A
) {
  return get_memory_usage(A.S);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  get_memory_usage_omm, unsigned long,
  const Hierarchical& A
) {
  unsigned long memory_usage = 0;
  for (int i=0; i<A.dim[0]; ++i) {
    for (int j=0; j<A.dim[1]; ++j) {
      memory_usage += get_memory_usage(A(i, j));
    }
  }
  return memory_usage;
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  get_memory_usage_omm, unsigned long,
  const UniformHierarchical& A
) {
  unsigned long memory_usage = 0;
  for (int i=0; i<A.dim[0]; ++i) {
    for (int j=0; j<A.dim[1]; ++j) {
      memory_usage += get_memory_usage(A(i, j));
    }
  }
  for (int i=0; i<A.dim[0]; ++i) {
    memory_usage += get_memory_usage(A.get_row_basis(i));
  }
  for (int j=0; j<A.dim[1]; ++j) {
    memory_usage += get_memory_usage(A.get_col_basis(j));
  }
  return memory_usage;
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  get_memory_usage_omm, unsigned long,
  const Node& A
) {
  std::cerr << "get_memory_usage(";
  std::cerr << A.type();
  std::cerr << ") undefined." << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma
