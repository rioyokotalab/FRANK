#include "hicma/util/initialize.h"

#include "hicma/classes.h"

#include "yorel/yomm2/cute.hpp"


namespace hicma
{

// Register all classes for the open multi methods
register_class(Node)
register_class(Dense, Node)
register_class(LowRank, Node)
register_class(LowRankShared, Node)
register_class(Hierarchical, Node)
register_class(NoCopySplit, Hierarchical)
register_class(UniformHierarchical, Hierarchical)

void initialize() {
  // Update virtual tables for open multi methods
  yorel::yomm2::update_methods();
}

} // namespace hicma
