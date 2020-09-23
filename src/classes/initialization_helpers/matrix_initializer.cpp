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
  std::vector<IndexRange> adapted_ranges;
  for (const IndexRange& range : tracker.associated_ranges) {
    adapted_ranges.push_back({n_cols, range.n});
    n_cols += range.n;
  }
  Dense block_row(tracker.index_range.n, n_cols);
  std::vector<Dense> parts = block_row.split(
    {{0, tracker.index_range.n}}, adapted_ranges
  );
  uint64_t j = 0;
  for (const IndexRange& range : tracker.associated_ranges) {
    fill_dense_representation(parts[j++], tracker.index_range, range);
  }
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
  if (tracker.children.empty()) {
    Dense basis, _, __;
    // TODO For very small ranks the following line can cause problems! Only
    // happens with block_row, not with block_col. TSQR is better than small fat
    // QR it seems.
    int64_t sample_size = std::min(rank+5, block_row.dim[1]);
    std::tie(basis, _, __) = rsvd(block_row, sample_size);
    col_basis[tracker.index_range] = resize(basis, basis.dim[0], rank);
  } else {
    // Create slices of appropriate size
    // TODO Assumes same size of all subblocks
    Hierarchical block_rowH = split(block_row, tracker.children.size(), 1);
    Hierarchical nested_basis(tracker.children.size(), 1);
    Dense translation, _, __;
    for (uint64_t i=0; i<tracker.children.size(); ++i) {
      // Multiply transpose of subbases to the slices
      Dense compressed_block_row = gemm(
        col_basis[tracker.children[i].index_range], block_rowH[i],
        1, true, false
      );
      int64_t sample_size = std::min(rank+5, compressed_block_row.dim[0]);
      std::tie(translation, _, __) = rsvd(compressed_block_row, sample_size);
      nested_basis[i] = NestedBasis(
        col_basis[tracker.children[i].index_range],
        resize(translation, translation.dim[0], rank),
        true
      );
    }
    col_basis[tracker.index_range] = std::move(nested_basis);
  }
}

Dense MatrixInitializer::make_block_col(const NestedTracker& tracker) const {
  // Create col block without admissible blocks
  int64_t n_rows = 0;
  std::vector<IndexRange> adapted_ranges;
  for (const IndexRange& range : tracker.associated_ranges) {
    adapted_ranges.push_back({n_rows, range.n});
    n_rows += range.n;
  }
  Dense block_col(n_rows, tracker.index_range.n);
  std::vector<Dense> parts = block_col.split(
    adapted_ranges, {{0, tracker.index_range.n}}
  );
  int64_t i = 0;
  for (const IndexRange& range : tracker.associated_ranges) {
    fill_dense_representation(parts[i++], range, tracker.index_range);
  }
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
  if (tracker.children.empty()) {
    Dense _, __, basis;
    int64_t sample_size = std::min(rank+5, block_col.dim[1]);
    std::tie(_, __, basis) = rsvd(block_col, sample_size);
    row_basis[tracker.index_range] = resize(basis, rank, basis.dim[1]);
  } else {
    // Create slices of appropriate size
    // NOTE Assumes same size of all subblocks
    Hierarchical block_colH = split(block_col, 1, tracker.children.size());
    Hierarchical nested_basis(1, tracker.children.size());
    Dense _, __, translation;
    for (uint64_t j=0; j < tracker.children.size(); ++j) {
      // Multiply transpose of subbases to the slices
      Dense compressed_block_col = gemm(
        block_colH[j], row_basis[tracker.children[j].index_range],
        1, false, true
      );
      int64_t sample_size = std::min(rank+5, compressed_block_col.dim[1]);
      std::tie(_, __, translation) = rsvd(compressed_block_col, sample_size);
      row_basis[tracker.index_range] = NestedBasis(
        row_basis[tracker.children[j].index_range],
        resize(translation, rank, translation.dim[1]),
        false
      );
    }
  }
}

} // namespace hicma
