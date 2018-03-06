#include "node.h"
#include "hierarchical.h"

namespace hicma {
  Node::Node() {
    i_abs = 0;
    j_abs = 0;
    level = 0;
  }

  Node::Node(const Hierarchical* parent, const int i_rel, const int j_rel) {
    if (parent) {
      i_abs = (parent->i_abs << 1) + i_rel;
      j_abs = (parent->j_abs << 1) + j_rel;
      level = parent->level + 1;
    }
    else {
      i_abs = 0;
      j_abs = 0;
      level = 0;
    }
  };
}
