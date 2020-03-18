#ifndef hicma_operations_LAPACK_qr_h
#define hicma_operations_LAPACK_qr_h

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/dense.h"

#include <tuple>

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

namespace hicma
{

  typedef std::tuple<Dense, Dense> dense_tuple;

  void qr(Node&, Node&, Node&);

  bool need_split(const Node&);

  std::tuple<Dense, Dense> make_left_orthogonal(const Node&);

  void update_splitted_size(const Node&, int&, int&);

  NodeProxy split_by_column(const Node&, Node&, int&);

  NodeProxy concat_columns(const Node&, const Node&, const Node&, int&);

  declare_method(
    void, qr_omm,
    (virtual_<Node&>, virtual_<Node&>, virtual_<Node&>)
  );
  declare_method(bool, need_split_omm, (virtual_<const Node&>));
  declare_method(
    dense_tuple, make_left_orthogonal_omm, (virtual_<const Node&>));
  declare_method(
    void, update_splitted_size_omm, (virtual_<const Node&>, int&, int&));
  declare_method(
    NodeProxy, split_by_column_omm,
    (virtual_<const Node&>, virtual_<Node&>, int&)
  );
  declare_method(
    NodeProxy, concat_columns_omm,
    (virtual_<const Node&>, virtual_<const Node&>, virtual_<const Node&>, int&)
  );

  void zero_lowtri(Node&);

  void zero_whole(Node&);
  declare_method(void, zero_lowtri_omm, (virtual_<Node&>));
  declare_method(void, zero_whole_omm, (virtual_<Node&>));


  void rq(Node&, Node&, Node&);

  declare_method(
    void, rq_omm, (virtual_<Node&>, virtual_<Node&>, virtual_<Node&>));

} // namespace hicma

#endif // hicma_operations_LAPACK_qr_h
