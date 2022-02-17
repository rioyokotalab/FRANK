#include "hicma/classes/initialization_helpers/matrix_initializer.h"

#include "hicma/classes/initialization_helpers/cluster_tree.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"

#include <iostream>


namespace hicma
{

// explicit template initialization (these are the only available types)
template Dense<float> MatrixInitializer::get_dense_representation(const ClusterTree&) const;
template Dense<double> MatrixInitializer::get_dense_representation(const ClusterTree&) const;
template LowRank<float> MatrixInitializer::get_compressed_representation(const ClusterTree&) const;
template LowRank<double> MatrixInitializer::get_compressed_representation(const ClusterTree&) const;

MatrixInitializer::MatrixInitializer(
    double admis, int64_t rank, int admis_type) : admis(admis),
    rank(rank), admis_type(admis_type) {}

template<typename T>
Dense<T> MatrixInitializer::get_dense_representation(
  const ClusterTree& node) const {
    Dense<T> representation(node.rows.n, node.cols.n);
    fill_dense_representation(representation, node.rows, node.cols);
    return representation;
  }

template<typename T>
LowRank<T> MatrixInitializer::get_compressed_representation(
  const ClusterTree& node) const {
    // TODO This function still relies on ClusterTree to be symmetric!
    return LowRank<T>(get_dense_representation<T>(node), rank);
  }

bool MatrixInitializer::is_admissible(const ClusterTree& node) const {
  bool admissible = true;
  // Vectors are never admissible
  admissible &= (node.rows.n > 1 && node.cols.n > 1);
  if(admis_type == POSITION_BASED_ADMIS) {
    admissible &= (node.dist_to_diag() > (int64_t)admis);
    return admissible;
  }
  // TODO improve error handling
  else {
    std::cerr<<"Only position-based admissibility is implemented for this type of construction"<<std::endl;
    std::abort();
  }
}

} // namespace hicma
