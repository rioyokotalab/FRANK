#include "hicma/classes/no_copy_split.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/index_range.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/operations/misc/get_dim.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"

#include <memory>
#include <utility>


namespace hicma
{

std::unique_ptr<Node> NoCopySplit::clone() const {
  return std::make_unique<NoCopySplit>(*this);
}

std::unique_ptr<Node> NoCopySplit::move_clone() {
  return std::make_unique<NoCopySplit>(std::move(*this));
}

const char* NoCopySplit::type() const { return "NoCopySplit"; }

NoCopySplit::NoCopySplit(Node& A, int ni_level, int nj_level) {
  dim[0] = ni_level; dim[1] = nj_level;
  row_range = IndexRange(0, get_n_rows(A));
  col_range = IndexRange(0, get_n_cols(A));
  create_children();
  for (int i=0; i<dim[0]; ++i) {
    for (int j=0; j<dim[1]; ++j) {
      (*this)(i, j) = make_view(row_range[i], col_range[j], A);
    }
  }
}

NoCopySplit::NoCopySplit(Node& A, const Hierarchical& like) {
  dim[0] = like.dim[0]; dim[1] = like.dim[1];
  row_range = IndexRange(0, get_n_rows(A));
  col_range = IndexRange(0, get_n_cols(A));
  create_children();
  int row_begin = 0;
  for (int i=0; i<dim[0]; ++i) {
    int col_begin = 0;
    for (int j=0; j<dim[1]; ++j) {
      (*this)(i, j) = make_view(
        IndexRange(row_begin, get_n_rows(like(i, j))),
        IndexRange(col_begin, get_n_cols(like(i, j))),
        A
      );
      col_begin += get_n_cols(like(i, j));
    }
    row_begin += get_n_rows(like(i, 0));
  }
}

define_method(
  NodeProxy, make_view,
  (const IndexRange& row_range, const IndexRange& col_range, Dense& A)
) {
  return Dense(row_range, col_range, A);
}

define_method(
  NodeProxy, make_view,
  (
    [[maybe_unused]] const IndexRange&, [[maybe_unused]] const IndexRange&,
    Node& A
  )
) {
  omm_error_handler("make_view", {A}, __FILE__, __LINE__);
  abort();
}

NoCopySplit::NoCopySplit(const Node& A, int ni_level, int nj_level) {
  dim[0] = ni_level; dim[1] = nj_level;
  row_range = IndexRange(0, get_n_rows(A));
  col_range = IndexRange(0, get_n_cols(A));
  create_children();
  for (int i=0; i<dim[0]; ++i) {
    for (int j=0; j<dim[1]; ++j) {
      (*this)(i, j) = make_view(row_range[i], col_range[j], A);
    }
  }
}

NoCopySplit::NoCopySplit(const Node& A, const Hierarchical& like) {
  dim[0] = like.dim[0]; dim[1] = like.dim[1];
  row_range = IndexRange(0, get_n_rows(A));
  col_range = IndexRange(0, get_n_cols(A));
  create_children();
  int row_begin = 0;
  for (int i=0; i<dim[0]; ++i) {
    int col_begin = 0;
    for (int j=0; j<dim[1]; ++j) {
      (*this)(i, j) = make_view(
        IndexRange(row_begin, get_n_rows(like(i, j))),
        IndexRange(col_begin, get_n_cols(like(i, j))),
        A
      );
      col_begin += get_n_cols(like(i, j));
    }
    row_begin += get_n_rows(like(i, 0));
  }
}

define_method(
  NodeProxy, make_view,
  (const IndexRange& row_range, const IndexRange& col_range, const Dense& A)
) {
  return Dense(row_range, col_range, A);
}

define_method(
  NodeProxy, make_view,
  (const IndexRange& row_range, const IndexRange& col_range, const LowRank& A)
) {
  return LowRank(row_range, col_range, A);
}

define_method(
  NodeProxy, make_view,
  (
    [[maybe_unused]] const IndexRange&, [[maybe_unused]] const IndexRange&,
    const Node& A
  )
) {
  omm_error_handler("make_view (const)", {A}, __FILE__, __LINE__);
  abort();
}

} // namespace hicma
