#include "hicma/util/initialize.h"
#include "hicma/classes.h"

#include "yorel/yomm2/cute.hpp"


namespace hicma
{

// Register all classes for the open multi methods
register_class(Matrix)
register_class(Dense<double>, Matrix)
register_class(Dense<float>, Matrix)
register_class(Empty, Matrix)
register_class(LowRank<double>, Matrix)
register_class(Hierarchical<double>, Matrix)

class Runtime {
 public:
  Runtime() {}

  ~Runtime() {}

  void start() {
    // Update virtual tables for open multi methods
    yorel::yomm2::update_methods();
  }

};

static Runtime runtime;

void initialize() {
  runtime.start();
}

} // namespace hicma
