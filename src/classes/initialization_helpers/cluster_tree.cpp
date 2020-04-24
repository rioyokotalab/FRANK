#include "hicma/classes/intitialization_helpers/cluster_tree.h"

#include "hicma/classes/hierarchical.h"
#include "hicma/operations/misc.h"

#include <cmath>
#include <cstdint>
#include <functional>


namespace hicma
{

ClusterTree::ClusterTree(const ClusterTree& A)
: dim(A.dim), block_dim(A.block_dim), start(A.start), nleaf(A.nleaf),
  rel_pos(A.rel_pos), abs_pos(A.abs_pos), parent(nullptr), children(A.children)
{
  for (ClusterTree& child_node : children) {
    child_node.parent = this;
  }
}

ClusterTree::ClusterTree(
  int64_t n_rows, int64_t n_cols,
  int64_t n_row_blocks, int64_t n_col_blocks,
  int64_t i_start, int64_t j_start,
  int64_t nleaf,
  int64_t i_rel, int64_t j_rel,
  int64_t i_abs, int64_t j_abs,
  ClusterTree* parent
) : dim{n_rows, n_cols}, block_dim{n_row_blocks, n_col_blocks},
    start{i_start, j_start}, nleaf(nleaf), rel_pos{i_rel, j_rel},
    abs_pos{i_abs, j_abs}, parent(parent)
{
  if (parent == nullptr || !is_leaf()) {
    children.reserve(block_dim[0]*block_dim[1]);
    for (int64_t i=0; i<block_dim[0]; ++i) {
      // Split row range
      int64_t child_n_rows = (dim[0]+block_dim[0]-1) / block_dim[0];
      int64_t child_row_start = start[0] + child_n_rows * i;
      if (i == block_dim[0]-1)
        child_n_rows = dim[0] - child_n_rows * (block_dim[0]-1);
      int64_t child_i_abs = abs_pos[0]*block_dim[0] + i;
      for (int64_t j=0; j<block_dim[1]; ++j) {
        // Split column range
        int64_t child_n_cols = (dim[1]+block_dim[1]-1) / block_dim[1];
        int64_t child_col_start = start[1] + child_n_cols * j;
        if (j == block_dim[1]-1)
          child_n_cols = dim[1] - child_n_cols * (block_dim[1]-1);
        int64_t child_j_abs = abs_pos[1]*block_dim[1] + j;
        children.emplace_back(
          child_n_rows, child_n_cols,
          n_row_blocks, n_col_blocks,
          child_row_start, child_col_start,
          nleaf,
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

const ClusterTree& ClusterTree::operator()(int64_t i, int64_t j) const {
  return children[i*block_dim[1] + j];
}

ClusterTree& ClusterTree::operator()(int64_t i, int64_t j) {
  return children[i*block_dim[1] + j];
}

ClusterTree::ClusterTree(
  const Hierarchical& like,
  int64_t i_start, int64_t j_start,
  int64_t i_rel, int64_t j_rel,
  int64_t i_abs, int64_t j_abs,
  ClusterTree* parent
) : dim{get_n_rows(like), get_n_cols(like)},
    block_dim{like.dim[0], like.dim[1]}, start{i_start, j_start},
    rel_pos{i_rel, j_rel}, abs_pos{i_abs, j_abs}, parent(parent)
{
  children.reserve(block_dim[0]*block_dim[1]);
  int64_t row_start = 0;
  for (int64_t i=0; i<block_dim[0]; ++i) {
    int64_t col_start = 0;
    int64_t n_rows = get_n_rows(like(i, 0));
    int64_t child_i_abs = abs_pos[0]*block_dim[0]+i;
    for (int64_t j=0; j<block_dim[1]; ++j) {
      int64_t n_cols = get_n_cols(like(i, j));
      int64_t child_j_abs = abs_pos[1]*block_dim[1]+j;
      children.emplace_back(
        n_rows, n_cols,
        0, 0,
        row_start, col_start,
        nleaf,
        i, j, child_i_abs, child_j_abs,
        this
      );
      col_start += n_cols;
    }
    row_start += n_rows;
  }
}

bool ClusterTree::is_leaf() const {
  bool leaf = true;
  leaf &= (dim[0] <= nleaf);
  leaf &= (dim[1] <= nleaf);
  // If leaf size is 0, consider any node a leaf
  leaf |= (nleaf == 0);
  return leaf;
}

int64_t ClusterTree::dist_to_diag() const {
  return std::abs(abs_pos[0] - abs_pos[1]);
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
