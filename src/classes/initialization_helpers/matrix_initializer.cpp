#include "hicma/classes/initialization_helpers/matrix_initializer.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/classes/initialization_helpers/basis_tracker.h"
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

MatrixInitializer::MatrixInitializer(
  int64_t admis, int64_t rank, int basis_type
) : admis(admis), rank(rank), basis_type(basis_type) {}

LowRank MatrixInitializer::get_compressed_representation(
  const ClusterTree& node
) {
  // TODO This function still relies on ClusterTree to be symmetric!
  LowRank out;
  if (basis_type == NORMAL_BASIS) {
    out = LowRank(get_dense_representation(node), rank);
  } else if (basis_type == SHARED_BASIS) {
    Dense D = get_dense_representation(node);
    Dense UtD = gemm(col_basis[node.rows], D, 1, true, false);
    Dense S = gemm(UtD, row_basis[node.cols], 1, false, true);
    out = LowRank(col_basis[node.rows], S, row_basis[node.cols]);
  }
  return out;
}

void MatrixInitializer::find_admissible_blocks(
  const ClusterTree& node
) {
  assert(basis_type == SHARED_BASIS);
  for (const ClusterTree& child : node) {
    if (is_admissible(child)) {
      col_tracker.register_range(child.cols, child.rows);
      row_tracker.register_range(child.rows, child.cols);
    } else {
      if (!child.is_leaf()) {
        find_admissible_blocks(child);
      }
    }
  }
}

void MatrixInitializer::create_nested_basis(const ClusterTree& node) {
  assert(basis_type == SHARED_BASIS);
  find_admissible_blocks(node);
  // NOTE Root level does not need basis, so loop over children
  for (NestedTracker& child : row_tracker.children) {
    construct_nested_col_basis(child);
  }
  for (NestedTracker& child : col_tracker.children) {
    construct_nested_row_basis(child);
  }
}

bool MatrixInitializer::is_admissible(const ClusterTree& node) const {
  bool admissible = true;
  // Main admissibility condition
  admissible &= (node.dist_to_diag() > admis);
  // Vectors are never admissible
  admissible &= (node.rows.n > 1 && node.cols.n > 1);
  return admissible;
}

Dense MatrixInitializer::make_block_row(const NestedTracker& tracker) const {
  // Create row block without admissible blocks
  int64_t n_cols = 0;
  for (const IndexRange& range : tracker.associated_ranges) {
    n_cols += range.n;
  }
  Dense block_row(tracker.index_range.n, n_cols);
  int64_t col_start = 0;
  // TODO Currently not working!
  abort();
  // for (const IndexRange& range : tracker.associated_ranges) {
  //   Dense part(block_row, tracker.index_range.n, range.n, 0, col_start);
  //   fill_dense_representation(part, tracker.index_range, range);
  //   col_start += range.n;
  // }
  return block_row;
}

void MatrixInitializer::construct_nested_col_basis(NestedTracker& tracker) {
  // If any children exist, complete the set of children so index range is
  // covered.
  if (!tracker.children.empty()) {
    // Gather uncovered index ranges
    tracker.complete_index_range();
    // Recursively make lower-level basis for these ranges
    for (NestedTracker& child : tracker.children) {
      construct_nested_col_basis(child);
    }
  }
  // Make block row using associated ranges
  Dense block_row = make_block_row(tracker);
  // If there are children, use the now complete set to update the block row
  // NOTE This assumes constant rank!
  if (!tracker.children.empty()) {
    Dense compressed_block_row(tracker.children.size()*rank, block_row.dim[1]);
    // Created slices of appropriate size
    // NOTE Assumes same size of all subblocks
    Hierarchical block_rowH = split(block_row, tracker.children.size(), 1);
    Hierarchical compressed_block_rowH = split(
      compressed_block_row, tracker.children.size(), 1
    );
    // Multiply transpose of subbases to the slices
    for (uint64_t i=0; i < tracker.children.size(); ++i) {
      gemm(
        col_basis[tracker.children[i].index_range], block_rowH[i],
        compressed_block_rowH[i],
        true, false, 1, 0
      );
    }
    // Replace block row with its compressed form
    block_row = std::move(compressed_block_row);
  }
  // Get column basis of (possibly compressed) block row
  int64_t sample_size = std::min(rank+5, tracker.index_range.n);
  Dense U, _, __;
  // TODO For very small ranks the following line can cause problems! Only
  // happens with block_row, not with block_col. TSQR is better than small fat
  // QR it seems.
  std::tie(U, _, __) = rsvd(block_row, sample_size);
  U = resize(U, U.dim[0], rank);
  std::vector<MatrixProxy> child_bases;
  for (NestedTracker& child : tracker.children) {
    child_bases.push_back(share_basis(col_basis[child.index_range]));
  }
  col_basis[tracker.index_range] = NestedBasis(std::move(U), child_bases, true);
}

Dense MatrixInitializer::make_block_col(const NestedTracker& tracker) const {
  // Create col block without admissible blocks
  int64_t n_rows = 0;
  for (const IndexRange& range : tracker.associated_ranges) {
    n_rows += range.n;
  }
  Dense block_col(n_rows, tracker.index_range.n);
  int64_t row_start = 0;
  // TODO Currently not working!
  abort();
  // for (const IndexRange& range : tracker.associated_ranges) {
  //   Dense part(block_col, range.n, tracker.index_range.n, row_start, 0);
  //   fill_dense_representation(part, range, tracker.index_range);
  //   row_start += range.n;
  // }
  return block_col;
}

void MatrixInitializer::construct_nested_row_basis(NestedTracker& tracker) {
  // If any children exist, complete the set of children so index range is
  // covered.
  if (!tracker.children.empty()) {
    // Gather uncovered index ranges
    tracker.complete_index_range();
    // Recursively make lower-level basis for these ranges
    for (NestedTracker& child : tracker.children) {
      construct_nested_row_basis(child);
    }
  }
  // Make block col using associated ranges
  Dense block_col = make_block_col(tracker);
  // If there are children, use the now complete set to update the block row
  // NOTE This assumes constant rank!
  if (!tracker.children.empty()) {
    Dense compressed_block_col(block_col.dim[0], tracker.children.size()*rank);
    // Created slices of appropriate size
    // NOTE Assumes same size of all subblocks
    Hierarchical block_colH = split(block_col, 1, tracker.children.size());
    Hierarchical compressed_block_colH = split(
      compressed_block_col, 1, tracker.children.size()
    );
    // Multiply transpose of subbases to the slices
    for (uint64_t j=0; j < tracker.children.size(); ++j) {
      gemm(
        block_colH[j], row_basis[tracker.children[j].index_range],
        compressed_block_colH[j],
        false, true, 1, 0
      );
    }
    // Replace block row with its compressed form
    block_col = std::move(compressed_block_col);
  }
  // Get column basis of (possibly compressed) block row
  int64_t sample_size = std::min(rank+5, tracker.index_range.n);
  Dense _, __, V;
  std::tie(_, __, V) = rsvd(block_col, sample_size);
  V = resize(V, rank, V.dim[1]);
  std::vector<MatrixProxy> child_bases;
  for (NestedTracker& child : tracker.children) {
    child_bases.push_back(share_basis(row_basis[child.index_range]));
  }
  row_basis[tracker.index_range] = NestedBasis(std::move(V), child_bases, false);
}

} // namespace hicma
