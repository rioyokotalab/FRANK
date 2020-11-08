#include "hicma/classes/initialization_helpers/matrix_initializer.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/initialization_helpers/cluster_tree.h"
#include "hicma/functions.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/misc.h"
#include "hicma/operations/randomized_factorizations.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>


namespace hicma
{

MatrixInitializer::MatrixInitializer(int64_t admis, int64_t rank)
: admis(admis), rank(rank) {}

Dense MatrixInitializer::get_dense_representation(
  const ClusterTree& node
) const {
  Dense representation(node.rows.n, node.cols.n);
  fill_dense_representation(representation, node.rows, node.cols);
  return representation;
}

LowRank MatrixInitializer::get_compressed_representation(
  const ClusterTree& node
) {
  // TODO This function still relies on ClusterTree to be symmetric!
  return LowRank(get_dense_representation(node), rank);
}

bool MatrixInitializer::is_admissible(const ClusterTree& node) const {
  bool admissible = true;
  // Main admissibility condition
  admissible &= (node.dist_to_diag() > admis);
  // Vectors are never admissible
  admissible &= (node.rows.n > 1 && node.cols.n > 1);
  return admissible;
}

} // namespace hicma
