#include "hicma/util/initialize.h"

#include "hicma/classes.h"
#include "hicma/classes/initialization_helpers/basis_tracker.h"
#include "hicma/util/pre_scheduler.h"

#include "starpu.h"
#include "yorel/yomm2/cute.hpp"


namespace hicma
{

// Register all classes for the open multi methods
register_class(Matrix)
register_class(Dense, Matrix)
register_class(LowRank, Matrix)
register_class(Hierarchical, Matrix)

void shutdown() {
  clear_trackers();
  starpu_shutdown();
}

class Runtime {
 public:
  Runtime() { initialize_starpu(); }

  ~Runtime() { shutdown(); }

  void start() {
  // Update virtual tables for open multi methods
    yorel::yomm2::update_methods();
  }
};

static Runtime runtime;

void initialize() { runtime.start(); }

} // namespace hicma
