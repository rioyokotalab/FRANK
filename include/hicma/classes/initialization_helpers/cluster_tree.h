/**
 * @file cluster_tree.h
 * @brief Include the `ClusterTree` class.
 *
 * @copyright Copyright (c) 2020
 */
#ifndef hicma_classes_initialization_helpers_cluster_tree_h
#define hicma_classes_initialization_helpers_cluster_tree_h

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

/**
 * @brief Tensor product of a row and column index tree
 *
 * @imageSize{cluster_tree.png,height:270px}
 * @image html cluster_tree.png "Illustration of a ClusterTree"
 *
 * An index tree is a hierarchical split of an index range.
 * Above figure illustrates how a cluster tree is formed by a tensor product
 * from these two index trees. Child nodes are thus analogous to submatrices or
 * subblocks.
 */
class ClusterTree {
 public:
  /**
   * @brief Row `IndexRange` associated with this `ClusterTree` node.
   */
  IndexRange rows;
  /**
   * @brief Column `IndexRange` associated with this `ClusterTree` node.
   */
  IndexRange cols;
  /**
   * @brief Number of splits of the {row, column} `IndexRange`s.
   */
  std::array<int64_t, 2> block_dim = {0, 0};
  /**
   * @brief Maximum size for leaf level submatrices.
   */
  int64_t nleaf = 0;
  /**
   * @brief Depth of this node in the `ClusterTree`.
   */
  int64_t level = 0;
  /**
   * @brief Relative position within immediate parent {block_row, block_col}
   */
  std::array<int64_t, 2> rel_pos = {0, 0};
  /**
   * @brief Absolute position on the level of this `ClusterTree` node
   * {block_row, block_col}
   *
   * The absolute position is calculated from that of the parent under the
   * assumption that all siblings of the parent have the same number of splits
   * of their block and row index ranges.
   */
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

  /**
   * @brief Construct a new `ClusterTree` object from `IndexRange` splits
   *
   * @param rows
   * Row `IndexRange` of the new `ClusterTree` node.
   * @param cols
   * Column `IndexRange` of the new `ClusterTree` node.
   * @param n_block_rows
   * Number of splits for the row `IndexRange` if leaf size isn't reached.
   * @param n_block_cols
   * Number of splits for the column `IndexRange` if leaf size isn't reached.
   * @param nleaf
   * Maximum size for leaf level `IndexRange`s.
   * @param level
   * Level of the new `ClusterTree` node within the `ClusterTree`.
   * @param i_rel
   * Relative position along the row splits within the parent `ClusterTree`
   * node.
   * @param j_rel
   * Relative position along the column splits within the parent `ClusterTree`
   * node.
   * @param i_abs
   * Absolute position along the row splits of the new `ClusterTree` node on its
   * level, assuming same splits across all siblings its parent node.
   * @param j_abs
   * Absolute position along the column splits of the new `ClusterTree` node on
   * its level, assuming same splits across all siblings its parent node.
   * @param parent
   * Pointer to the parent node of this new `ClusterTree` node. `nullptr` if the
   * it is to be the root.
   */
  ClusterTree(
    IndexRange rows, IndexRange cols,
    int64_t n_row_blocks=0, int64_t n_col_blocks=0,
    int64_t nleaf=0,
    int64_t level=0,
    int64_t i_rel=0, int64_t j_rel=0,
    int64_t i_abs=0, int64_t j_abs=0,
    ClusterTree* parent=nullptr
  );

  /**
   * @brief Return iterator to first child
   *
   * @return std::vector<ClusterTree>::iterator
   * Iterator initially set to the first child.
   *
   * The first child is the child whose row and column index ranges start on the
   * same index as that of its parent. When incremented, nodes will be returned
   * in the equivalent of row-major order.
   */
  std::vector<ClusterTree>::iterator begin();


  /**
   * @brief Return constant iterator to first child
   *
   * @return std::vector<ClusterTree>::iterator
   * Constant iterator initially set to the first child.
   *
   * The first child is the child whose row and column index ranges start on the
   * same index as that of its parent. When incremented, nodes will be returned
   * in the equivalent of row-major order.
   */
  std::vector<ClusterTree>::const_iterator begin() const;

  /**
   * @brief Return iterator just past the last child
   *
   * @return std::vector<ClusterTree>::iterator
   * Iterator just past the last child.
   *
   * The last child is the child whose row and column index ranges end on the
   * same index as that of its parent.
   */
  std::vector<ClusterTree>::iterator end();

  /**
   * @brief Return constant iterator just past the last child
   *
   * @return std::vector<ClusterTree>::iterator
   * Constant iterator just past the last child.
   *
   * The last child is the child whose row and column index ranges end on the
   * same index as that of its parent.
   */
  std::vector<ClusterTree>::const_iterator end() const;

  /**
   * @brief Return the child with the \p i-th row and \p-th column child index
   * range
   *
   * @param i
   * Child row index to be chosen.
   * @param j
   * Child column index to be chosen.
   * @return const ClusterTree&
   * Reference to the child with the \p i-th row and \p-th column child index
   * range.
   */
  ClusterTree& operator()(int64_t i, int64_t j);

  /**
   * @brief Return the child with the \p i-th row and \p-th column child index
   * range
   *
   * @param i
   * Child row index to be chosen.
   * @param j
   * Child column index to be chosen.
   * @return const ClusterTree&
   * Constant reference to the child with the \p i-th row and \p-th column child
   * index range .
   */
  const ClusterTree& operator()(int64_t i, int64_t j) const;

  /**
   * @brief Returns the distance to the diagonal in terms of blocks on the same
   * level
   *
   * @return int64_t
   * Distance to the diagonal
   *
   * The diagonal on any given level is defined as the blocks whose row and and
   * column index range have overlap.
   */
  int64_t dist_to_diag() const;

  /**
   * @brief Check if the `ClusterTree` node is a leaf
   *
   * @return true
   * If the node is a leaf and has no children.
   * @return false
   * If the node is not a leaf.
   */
  bool is_leaf() const;

  /**
   * @brief Get vector of all nodes on the same level that have the same row
   * index range
   *
   * @return std::vector<std::reference_wrapper<const ClusterTree>>
   * Block row defined by the `ClusterTree` nodes of the blocks
   */
  std::vector<std::reference_wrapper<const ClusterTree>> get_block_row() const;


  /**
   * @brief Get vector of all nodes on the same level that have the same column
   * index range
   *
   * @return std::vector<std::reference_wrapper<const ClusterTree>>
   * Block column defined by the `ClusterTree` nodes of the blocks
   */
  std::vector<std::reference_wrapper<const ClusterTree>> get_block_col() const;
};

} // namespace hicma


#endif // hicma_classes_initialization_helpers_cluster_tree_h
