#ifndef hicma_extension_headers_classes_h
#define hicma_extension_headers_classes_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class Node;
class Hierarchical;
class NoCopySplit;

MULTI_METHOD(
  make_hierarchical, Hierarchical,
  const virtual_<Node>&, int ni_level, int nj_level
);

MULTI_METHOD(
  make_no_copy_split, NoCopySplit,
  virtual_<Node>&, int ni_level, int nj_level
);

MULTI_METHOD(
  make_no_copy_split_const, NoCopySplit,
  const virtual_<Node>&, int ni_level, int nj_level
);

} // namespace hicma

#endif // hicma_extension_headers_classes_h
