#ifndef operations_qr_h
#define operations_qr_h

#include "hicma/node_proxy.h"
#include "hicma/node.h"

namespace hicma
{
  void qr(NodeProxy&, NodeProxy&, NodeProxy&);
  void qr(NodeProxy&, NodeProxy&, Node&);
  void qr(NodeProxy&, Node&, NodeProxy&);
  void qr(NodeProxy&, Node&, Node&);
  void qr(Node&, NodeProxy&, NodeProxy&);
  void qr(Node&, NodeProxy&, Node&);
  void qr(Node&, Node&, NodeProxy&);
  void qr(Node&, Node&, Node&);

  bool need_split(const NodeProxy&);
  bool need_split(const Node&);

  void make_left_orthogonal(const NodeProxy&, NodeProxy&, NodeProxy&);

  void update_splitted_size(const NodeProxy&, int&, int&);
  void update_splitted_size(const Node&, int&, int&);

  void split_by_column(const NodeProxy&, Node&, int&, NodeProxy&);
  void split_by_column(const Node&, Node&, int&, NodeProxy&);

  void concat_columns(const NodeProxy&, const Node&, NodeProxy&, int&, const NodeProxy&);
  void concat_columns(const Node&, const Node&, NodeProxy&, int&, const Node&);

} // namespace hicma


#endif // operations_geqrt_h
