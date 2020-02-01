#ifndef hicma_classes_no_cop_split_h
#define hicma_classes_no_cop_split_h

#include "hicma/classes/hierarchical.h"

#include <memory>

#include "yorel/multi_methods.hpp"

namespace hicma
{

class Node;

class NoCopySplit : public Hierarchical {
public:
  MM_CLASS(NoCopySplit, Hierarchical);
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

} // namespace hicma

#endif // hicma_classes_no_cop_split_h
