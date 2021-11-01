#include "hicma/util/initialize.h"

#include "hicma/classes.h"
#include "hicma/classes/initialization_helpers/basis_tracker.h"
#include "hicma/util/pre_scheduler.h"

//#include "starpu.h"
#include "yorel/yomm2/cute.hpp"


namespace hicma
{

// Register all classes for the open multi methods
register_class(Matrix)
register_class(Dense, Matrix)
register_class(Empty, Matrix)
register_class(LowRank, Matrix)
register_class(Hierarchical, Matrix)

class Runtime {
 private:
  bool init_starpu;
 public:
  Runtime() {}

  ~Runtime() {
    clear_trackers();
    //if(init_starpu) starpu_shutdown();
  }

  void start(bool starpu) {
    init_starpu = starpu;
    //if(init_starpu) initialize_starpu();
    // Update virtual tables for open multi methods
    yorel::yomm2::update_methods();
  }

};

static Runtime runtime;

void initialize(bool init_starpu) {
  runtime.start(init_starpu);
}

} // namespace hicma
