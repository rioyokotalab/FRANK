#ifndef hicma_classes_no_cop_split_h
#define hicma_classes_no_cop_split_h

#include "hicma/classes/hierarchical.h"

#include <memory>

#include "yorel/yomm2/cute.hpp"

namespace hicma
{

class Node;

class NoCopySplit : public Hierarchical {
public:
  // Special member functions
  NoCopySplit();

  ~NoCopySplit();

  NoCopySplit(const NoCopySplit& A);

  NoCopySplit& operator=(const NoCopySplit& A);

  NoCopySplit(NoCopySplit&& A);

  NoCopySplit& operator=(NoCopySplit&& A);

  // Overridden functions from Node
  std::unique_ptr<Node> clone() const override;

  std::unique_ptr<Node> move_clone() override;

  const char* type() const override;

  // Additional constructors
  NoCopySplit(Node&, int ni_level, int nj_level, bool node_only=false);

  NoCopySplit(const Node&, int ni_level, int nj_level, bool node_only=false);
};

register_class(NoCopySplit, Hierarchical);

} // namespace hicma

#endif // hicma_classes_no_cop_split_h
