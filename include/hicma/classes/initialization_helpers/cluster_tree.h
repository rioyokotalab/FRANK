#ifndef hicma_classes_initializion_helpers_cluster_tree_h
#define hicma_classes_initializion_helpers_cluster_tree_h

#include "hicma/classes/initialization_helpers/index_range.h"

#include <array>
#include <cstdint>
#include <functional>
#include <vector>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Hierarchical;

class ClusterTree {
 public:
  IndexRange rows, cols;
  std::array<int64_t, 2> block_dim = {0, 0};
  int64_t nleaf = 0;
  int64_t level = 0;
  std::array<int64_t, 2> rel_pos = {0, 0};
  std::array<int64_t, 2> abs_pos = {0, 0};
 private:
  ClusterTree* parent = nullptr;
  std::vector<ClusterTree> children;

 public:
  // Special member functions
  ClusterTree() = delete;

  ~ClusterTree() = default;

  ClusterTree(const ClusterTree& A) = delete;

  ClusterTree& operator=(const ClusterTree& A) = delete;

  ClusterTree(ClusterTree&& A) = default;

  ClusterTree& operator=(ClusterTree&& A) = default;

  // Additional constructors
  ClusterTree(
    IndexRange rows, IndexRange cols,
    int64_t n_row_blocks=0, int64_t n_col_blocks=0,
    int64_t nleaf=0,
    int64_t level=0,
    int64_t i_rel=0, int64_t j_rel=0,
    int64_t i_abs=0, int64_t j_abs=0,
    ClusterTree* parent=nullptr
  );

  ClusterTree(
    const Hierarchical& A,
    int64_t i_start=0, int64_t j_start=0,
    int64_t level=0,
    int64_t i_rel=0, int64_t j_rel=0,
    int64_t i_abs=0, int64_t j_abs=0,
    ClusterTree* parent=nullptr
  );

  // Make class usable as range
  std::vector<ClusterTree>::iterator begin();

  std::vector<ClusterTree>::const_iterator begin() const;

  std::vector<ClusterTree>::iterator end();

  std::vector<ClusterTree>::const_iterator end() const;

  // Child indexing
  const ClusterTree& operator()(int64_t i, int64_t j) const;

  ClusterTree& operator()(int64_t i, int64_t j);

  // Utility methods
  int64_t dist_to_diag() const;

  bool is_leaf() const;

  std::vector<std::reference_wrapper<const ClusterTree>> get_block_row() const;

  std::vector<std::reference_wrapper<const ClusterTree>> get_block_col() const;
};

} // namespace hicma


#endif // hicma_classes_initializion_helpers_cluster_tree_h
