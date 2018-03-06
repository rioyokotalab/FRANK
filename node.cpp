#include "node.h"
#include "hierarchical.h"

namespace hicma {
  Node::Node() {
    i = 0;
    j = 0;
    level = 0;
  }

  Node::Node(const Hierarchical* parent, const int i_rel, const int j_rel) {
    if (parent) {
      i = (parent->i << 1) + i_rel;
      j = (parent->j << 1) + j_rel;
      level = parent->level + 1;
    }
    else {
      i = 0;
      j = 0;
      level = 0;
    }
  };
}


