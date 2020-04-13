#ifndef hicma_extension_headers_operations_h
#define hicma_extension_headers_operations_h

#include "hicma/classes/dense.h"
#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <tuple>
#include <vector>


namespace hicma
{

// NOTE These typedefs are necessary since yomm macros use commas to parse the
// function signature, so type tuples cannot be defined.
typedef std::tuple<NodeProxy, NodeProxy> NodePair;
typedef std::tuple<Dense, Dense> DensePair;
typedef std::tuple<Dense, Dense, Dense> DenseTriplet;
typedef std::tuple<Dense, std::vector<int>> DenseIntVectorPair;

// Arithmetic
declare_method(
  Node&, addition_omm,
  (virtual_<Node&>, virtual_<const Node&>)
)

declare_method(
  NodeProxy, addition_omm,
  (virtual_<const Node&>, virtual_<const Node&>)
)

declare_method(
  NodeProxy, subtraction_omm,
  (virtual_<const Node&>, virtual_<const Node&>)
)

declare_method(
  Node&, multiplication_omm,
  (virtual_<Node&>, double)
)

// BLAS
declare_method(
  void, gemm_omm,
  (
    virtual_<const Node&>, virtual_<const Node&>, virtual_<Node&>,
    double, double
  )
)

declare_method(
  Dense, gemm_omm,
  (
    virtual_<const Node&>, virtual_<const Node&>,
    double, bool, bool
  )
)

declare_method(
  void, trmm_omm,
  (
    virtual_<const Node&>, virtual_<Node&>,
    const char&, const char&, const char&, const char&,
    const double&
  )
)

declare_method(
  void, trsm_omm,
  (virtual_<const Node&>, virtual_<Node&>, int, int)
)

// LAPACK
declare_method(DenseIntVectorPair, geqp3_omm, (virtual_<Node&>))

declare_method(
  void, geqrt_omm,
  (virtual_<Node&>, virtual_<Node&>)
)

declare_method(
  NodePair, getrf_omm,
  (virtual_<Node&>)
)

declare_method(DenseIntVectorPair, one_sided_id_omm, (virtual_<Node&>, int))

declare_method(DenseTriplet, id_omm, (virtual_<Node&>, int))

declare_method(
  void, larfb_omm,
  (virtual_<const Node&>, virtual_<const Node&>, virtual_<Node&>, bool)
)

declare_method(
  void, qr_omm,
  (virtual_<Node&>, virtual_<Node&>, virtual_<Node&>)
)
declare_method(bool, need_split_omm, (virtual_<const Node&>))
declare_method(
  DensePair, make_left_orthogonal_omm,
  (virtual_<const Node&>)
)
declare_method(
  void, update_splitted_size_omm,
  (virtual_<const Node&>, int&, int&)
)
declare_method(
  NodeProxy, split_by_column_omm,
  (virtual_<const Node&>, virtual_<Node&>, int&)
)
declare_method(
  NodeProxy, concat_columns_omm,
  (virtual_<const Node&>, virtual_<const Node&>, virtual_<const Node&>, int&)
)

declare_method(void, zero_lowtri_omm, (virtual_<Node&>))
declare_method(void, zero_whole_omm, (virtual_<Node&>))

declare_method(
  void, rq_omm,
  (virtual_<Node&>, virtual_<Node&>, virtual_<Node&>)
)

declare_method(
  void, tpmqrt_omm,
  (
    virtual_<const Node&>, virtual_<const Node&>,
    virtual_<Node&>, virtual_<Node&>,
    bool
  )
)

declare_method(
  void, tpqrt_omm,
  (virtual_<Node&>, virtual_<Node&>, virtual_<Node&>)
)

} // namespace hicma

#endif // hicma_extension_headers_operations_h
