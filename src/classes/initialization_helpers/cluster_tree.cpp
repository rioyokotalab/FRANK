#include "hicma/classes/initialization_helpers/cluster_tree.h"

#include <cmath>


namespace hicma
{

ClusterTree::ClusterTree(
  IndexRange rows, IndexRange cols,
  int64_t n_row_blocks, int64_t n_col_blocks,
  int64_t nleaf,
  int64_t level,
  int64_t i_rel, int64_t j_rel,
  int64_t i_abs, int64_t j_abs,
  ClusterTree* parent
) : rows(rows), cols(cols), block_dim{n_row_blocks, n_col_blocks},
    nleaf(nleaf), level(level), rel_pos{i_rel, j_rel},
    abs_pos{i_abs, j_abs}, parent(parent)
{
  if (parent == nullptr || !is_leaf()) {
    children.reserve(block_dim[0]*block_dim[1]);
    std::vector<IndexRange> row_subranges = rows.split(block_dim[0]);
    std::vector<IndexRange> col_subranges = cols.split(block_dim[1]);
    for (int64_t i=0; i<block_dim[0]; ++i) {
      int64_t child_i_abs = abs_pos[0]*block_dim[0] + i;
      for (int64_t j=0; j<block_dim[1]; ++j) {
        int64_t child_j_abs = abs_pos[1]*block_dim[1] + j;
        children.emplace_back(
          row_subranges[i], col_subranges[j],
          n_row_blocks, n_col_blocks,
          nleaf,
          level+1,
          i, j, child_i_abs, child_j_abs,
          this
        );
      }
    }
  }
}

std::vector<ClusterTree>::iterator ClusterTree::begin() {
  return children.begin();
}

std::vector<ClusterTree>::const_iterator ClusterTree::begin() const {
  return children.begin();
}

std::vector<ClusterTree>::iterator ClusterTree::end() {
  return children.end();
}

std::vector<ClusterTree>::const_iterator ClusterTree::end() const {
  return children.end();
}

ClusterTree& ClusterTree::operator()(int64_t i, int64_t j) {
  return children[i*block_dim[1] + j];
}

const ClusterTree& ClusterTree::operator()(int64_t i, int64_t j) const {
  return children[i*block_dim[1] + j];
}

int64_t ClusterTree::dist_to_diag() const {
  return std::abs(abs_pos[0] - abs_pos[1]);
}

bool ClusterTree::is_leaf() const {
  bool leaf = true;
  leaf &= (rows.n <= nleaf);
  leaf &= (cols.n <= nleaf);
  // If leaf size is 0, consider any node a leaf
  leaf |= (nleaf == 0);
  return leaf;
}

std::vector<std::reference_wrapper<const ClusterTree>>
ClusterTree::get_block_row() const {
  std::vector<std::reference_wrapper<const ClusterTree>> block_row;
  for (int64_t j=0; j<parent->block_dim[1]; ++j) {
    block_row.emplace_back((*parent)(rel_pos[0], j));
  }
  return block_row;
}

std::vector<std::reference_wrapper<const ClusterTree>>
ClusterTree::get_block_col() const {
  std::vector<std::reference_wrapper<const ClusterTree>> block_col;
  for (int64_t i=0; i<parent->block_dim[0]; ++i) {
    block_col.emplace_back((*parent)(i, rel_pos[1]));
  }
  return block_col;
}

} // namespace hicma
