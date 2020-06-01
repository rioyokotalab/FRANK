#include "hicma/classes/intitialization_helpers/matrix_initializer.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/shared_basis.h"
#include "hicma/classes/intitialization_helpers/cluster_tree.h"
#include "hicma/operations/BLAS.h"

#include <cstdint>


namespace hicma
{

MatrixInitializer::MatrixInitializer(
  void (*kernel)(
    Dense& A,
    const std::vector<std::vector<double>>& x,
    int64_t row_start, int64_t col_start
  ),
  const std::vector<std::vector<double>>& x,
  int64_t admis, int64_t rank,
  int basis_type
) : kernel(kernel), x(x), admis(admis), rank(rank), basis_type(basis_type) {}

void MatrixInitializer::fill_dense_representation(
  Dense& A,
  const ClusterTree& node
) const {
  kernel(A, x, node.start[0], node.start[1]);
}

Dense MatrixInitializer::get_dense_representation(
  const ClusterTree& node
) const {
  Dense representation(node.dim[0], node.dim[1]);
  kernel(representation, x, node.start[0], node.start[1]);
  return representation;
}

Dense MatrixInitializer::make_block_row(const ClusterTree& node) const {
  // Create row block without admissible blocks
  int64_t n_cols = 0;
  std::vector<std::reference_wrapper<const ClusterTree>> admissible_blocks;
  for (const ClusterTree& block : node.get_block_row()) {
    if (is_admissible(block)) {
      admissible_blocks.emplace_back(block);
      n_cols += block.dim[1];
    }
  }
  Dense block_row(node.dim[0], n_cols);
  int64_t col_start = 0;
  for (const ClusterTree& block : admissible_blocks) {
    Dense part(block_row, block.dim[0], block.dim[1], 0, col_start);
    fill_dense_representation(part, block);
    col_start += block.dim[1];
  }
  return block_row;
}

Dense MatrixInitializer::make_block_col(const ClusterTree& node) const {
  // Create col block without admissible blocks
  int64_t n_rows = 0;
  std::vector<std::reference_wrapper<const ClusterTree>> admissible_blocks;
  for (const ClusterTree& block : node.get_block_col()) {
    if (is_admissible(block)) {
      admissible_blocks.emplace_back(block);
      n_rows += block.dim[0];
    }
  }
  Dense block_col(n_rows, node.dim[1]);
  int64_t row_start = 0;
  for (const ClusterTree& block : admissible_blocks) {
    Dense part(block_col, block.dim[0], block.dim[1], row_start, 0);
    fill_dense_representation(part, block);
    row_start += block.dim[0];
  }
  return block_col;
}

LowRank MatrixInitializer::get_compressed_representation(
  const ClusterTree& node
) {
  LowRank out;
  if (basis_type == NORMAL_BASIS) {
    out = LowRank(get_dense_representation(node), rank);
  } else if (basis_type == SHARED_BASIS) {
    if (col_bases.find({node.level, node.abs_pos[0]}) == col_bases.end()) {
      Dense row_block = make_block_row(node);
      // TODO: The following line is probably copying right now as there is now
      // Dense(Matrix&&) or Dense(MatrixProxy&&) constructor.
      // TODO Consider making a unified syntax for OMM constructors.
      col_bases[{node.level, node.abs_pos[0]}] = std::make_shared<Dense>(
        std::move(LowRank(row_block, rank).U));
    }
    if (row_bases.find({node.level, node.abs_pos[1]}) == row_bases.end()) {
      Dense col_block = make_block_col(node);
      row_bases[{node.level, node.abs_pos[1]}] = std::make_shared<Dense>(
        std::move(LowRank(col_block, rank).V));
    }
    Dense D = get_dense_representation(node);
    Dense S = gemm(
      gemm(*col_bases.at({node.level, node.abs_pos[0]}), D, 1, true, false),
      *row_bases.at({node.level, node.abs_pos[1]}),
      1, false, true
    );
    out = LowRank(
      SharedBasis(col_bases.at({node.level, node.abs_pos[0]})),
      S,
      SharedBasis(row_bases.at({node.level, node.abs_pos[1]}))
    );
  }
  return out;
}

bool MatrixInitializer::is_admissible(const ClusterTree& node) const {
  bool admissible = true;
  // Main admissibility condition
  admissible &= (node.dist_to_diag() > admis);
  // Vectors are never admissible
  admissible &= (node.dim[0] > 1 && node.dim[1] > 1);
  return admissible;
}

} // namespace hicma
