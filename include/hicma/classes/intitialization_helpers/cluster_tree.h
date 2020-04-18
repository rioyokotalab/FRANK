#ifndef hicma_classes_initializion_helpers_cluster_tree_h
#define hicma_classes_initializion_helpers_cluster_tree_h

#include <array>
#include <cstdint>
#include <vector>


namespace hicma
{

class Hierarchical;

class ClusterTree {
 private:
  ClusterTree* parent = nullptr;
 public:
  std::array<int64_t, 2> dim = {0, 0};
  std::array<int64_t, 2> begin = {0, 0};
  std::array<int64_t, 2> rel_pos = {0, 0};
  std::array<int64_t, 2> abs_pos = {0, 0};
  std::vector<ClusterTree> children;
  std::array<int64_t, 2> block_dim = {0, 0};

  // Special member functions
  ClusterTree() = delete;

  ~ClusterTree() = default;

  ClusterTree(const ClusterTree& A);

  ClusterTree& operator=(const ClusterTree& A) = delete;

  ClusterTree(ClusterTree&& A) = delete;

  ClusterTree& operator=(ClusterTree&& A) = default;

  // Additional constructors
  ClusterTree(
    int64_t n_rows, int64_t n_cols,
    int64_t i_begin=0, int64_t j_begin=0,
    int64_t i_rel=0, int64_t j_rel=0,
    int64_t i_abs=0, int64_t j_abs=0,
    ClusterTree* parent=nullptr
  );

  // Utility methods
  void split(int64_t n_row_blocks, int64_t n_col_blocks);

  void split(const Hierarchical& like);

  std::vector<const ClusterTree*> get_block_row() const;

  std::vector<const ClusterTree*> get_block_col() const;
};

} // namespace hicma


#endif // hicma_classes_initializion_helpers_cluster_tree_h
