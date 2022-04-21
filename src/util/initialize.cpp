#include "FRANK/util/initialize.h"
#include "FRANK/classes.h"

#include "yorel/yomm2/cute.hpp"


namespace FRANK {
  // Register all classes for the open multi methods
  register_class(Matrix)
  register_class(Dense, Matrix)
  register_class(Empty, Matrix)
  register_class(LowRank, Matrix)
  register_class(Hierarchical, Matrix)

  void initialize() {
    yorel::yomm2::update_methods();
  }

} // namespace FRANK
