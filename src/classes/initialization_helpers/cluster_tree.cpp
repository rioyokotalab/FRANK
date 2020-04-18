#include "hicma/classes/intitialization_helpers/cluster_tree.h"

#include "hicma/classes/hierarchical.h"
#include "hicma/operations/misc/get_dim.h"

#include <cstdint>


namespace hicma
{

ClusterTree::ClusterTree(const ClusterTree& A)
: parent(nullptr), dim(A.dim), begin(A.begin),
  rel_pos(A.rel_pos), abs_pos(A.abs_pos),
  children(A.children), block_dim(A.block_dim)
{
  for (ClusterTree& child_node : children) {
    child_node.parent = this;
  }
}

ClusterTree::ClusterTree(
  int64_t n_rows, int64_t n_cols,
  int64_t i_begin, int64_t j_begin,
  int64_t i_rel, int64_t j_rel,
  int64_t i_abs, int64_t j_abs,
  ClusterTree* parent
) : parent(parent), dim{n_rows, n_cols}, begin{i_begin, j_begin},
    rel_pos{i_rel, j_rel}, abs_pos{i_abs, j_abs}
{}

void ClusterTree::split(
  int64_t n_row_blocks, int64_t n_col_blocks
) {
  block_dim = {n_row_blocks, n_col_blocks};
  children.reserve(block_dim[0]*block_dim[1]);
  for (int64_t i=0; i<block_dim[0]; ++i) {
    // Split row range
    int64_t child_n_rows = (dim[0]+block_dim[0]-1) / block_dim[0];
    int64_t child_row_begin = begin[0] + child_n_rows * i;
    if (i == block_dim[0]-1)
      child_n_rows = dim[0] - child_n_rows * (block_dim[0]-1);
    int64_t child_i_abs = abs_pos[0]*block_dim[0] + i;
    for (int64_t j=0; j<block_dim[1]; ++j) {
      // Split column range
      int64_t child_n_cols = (dim[1]+block_dim[1]-1) / block_dim[1];
      int64_t child_col_begin = begin[1] + child_n_cols * j;
      if (j == block_dim[1]-1)
        child_n_cols = dim[1] - child_n_cols * (block_dim[1]-1);
      int64_t child_j_abs = abs_pos[1]*block_dim[1] + j;
      children.emplace_back(
        child_n_rows, child_n_cols,
        child_row_begin, child_col_begin,
        i, j, child_i_abs, child_j_abs,
        this
      );
    }
  }
}

void ClusterTree::split(const Hierarchical& like) {
  block_dim = {like.dim[0], like.dim[1]};
  children.reserve(block_dim[0]*block_dim[1]);
  int64_t row_begin = 0;
  for (int64_t i=0; i<block_dim[0]; ++i) {
    int64_t col_begin = 0;
    int64_t n_rows = get_n_rows(like(i, 0));
    int64_t child_i_abs = abs_pos[0]*block_dim[0]+i;
    for (int64_t j=0; j<block_dim[1]; ++j) {
      int64_t n_cols = get_n_cols(like(i, j));
      int64_t child_j_abs = abs_pos[1]*block_dim[1]+j;
      children.emplace_back(
        n_rows, n_cols,
        row_begin, col_begin,
        i, j, child_i_abs, child_j_abs,
        this
      );
      col_begin += n_cols;
    }
    row_begin += n_rows;
  }
}

std::vector<const ClusterTree*> ClusterTree::get_block_row() const {
  std::vector<const ClusterTree*> block_row(parent->block_dim[1]);
  for (int64_t j=0; j<parent->block_dim[1]; ++j) {
    block_row[j] = &(parent->children[rel_pos[0]*parent->block_dim[1]+j]);
  }
  return block_row;
}

std::vector<const ClusterTree*> ClusterTree::get_block_col() const {
  std::vector<const ClusterTree*> block_col(parent->block_dim[0]);
  for (int64_t i=0; i<parent->block_dim[0]; ++i) {
    block_col[i] = &(parent->children[i*parent->block_dim[1]+rel_pos[1]]);
  }
  return block_col;
}

} // namespace hicma
