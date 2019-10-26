#ifndef operations_qr_h
#define operations_qr_h

#include <tuple>

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{
  class Node;
  class Dense;

  typedef std::tuple<Dense, Dense> dense_tuple;

  void qr(Node&, Node&, Node&);

  bool need_split(const Node&);

  std::tuple<Dense, Dense> make_left_orthogonal(const Node&);

  void update_splitted_size(const Node&, int&, int&);

  void split_by_column(const Node&, Node&, int&, Node&);

  void concat_columns(const Node&, const Node&, Node&, int&, const Node&);

  MULTI_METHOD(
    qr_omm, void,
    virtual_<Node>&, virtual_<Node>&, virtual_<Node>&
  );
  MULTI_METHOD(
    need_split_omm, bool,
    const virtual_<Node>&
  );
  MULTI_METHOD(
    make_left_orthogonal_omm, dense_tuple,
    const virtual_<Node>&
  );
  MULTI_METHOD(
    update_splitted_size_omm, void,
    const virtual_<Node>&, int&, int&
  );
  MULTI_METHOD(
    split_by_column_omm, void,
    const virtual_<Node>&, virtual_<Node>&,
    int&, Node&
  );
  MULTI_METHOD(
    concat_columns_omm, void,
    const virtual_<Node>&, const virtual_<Node>&, Node&,
    int&, const virtual_<Node>&
  );

} // namespace hicma

#endif // operations_geqrt_h
