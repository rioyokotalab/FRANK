#include "hicma/util/pre_scheduler.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/initialization_helpers/basis_tracker.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"

#include "hicma_private/starpu_data_handler.h"

#include "starpu.h"
#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>


namespace hicma
{

class Task {
 public:
  // TODO Remove these and let tasks have individual arguments!
  std::vector<Dense> constant;
  std::vector<Dense> modified;
  starpu_task* task;

  // Special member functions
  Task() = default;

  virtual ~Task() = default;

  Task(const Task& A) = default;

  Task& operator=(const Task& A) = default;

  Task(Task&& A) = default;

  Task& operator=(Task&& A) = default;

  // Execute the task
  virtual void submit() = 0;

 protected:
  Task(
    std::vector<std::reference_wrapper<const Dense>> constant_,
    std::vector<std::reference_wrapper<Dense>> modified_
  ) {
    for (size_t i=0; i<constant_.size(); ++i) {
      constant.push_back(constant_[i].get().shallow_copy());
    }
    for (size_t i=0; i<modified_.size(); ++i) {
      modified.push_back(modified_[i].get().shallow_copy());
    }
  }

  starpu_data_handle_t get_handle(const Dense& A) {
    return A.data->get_handle();
  }

  DataHandler& get_handler(const Dense& A) { return *A.data;}
};

std::list<std::shared_ptr<Task>> tasks;
bool schedule_started = false;
bool is_tracking = false;

void add_task(std::shared_ptr<Task> task) {
  if (schedule_started) {
    tasks.push_back(task);
  } else {
    task->submit();
  }
}


void initialize_starpu() {
  STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "init");
  //make_copy_codelet();
  //make_transpose_codelet();
  //make_assign_codelet();
  //make_addition_codelet();
  //make_subtraction_codelet();
  //make_multiplication_codelet();
  //make_getrf_codelet();
  //make_qr_codelet();
  //make_rq_codelet();
  //make_trsm_codelet();
  //make_gemm_codelet();
  //make_svd_codelet();
}

void start_tracking() {
  assert(!is_tracking);
  is_tracking = true;
}

void stop_tracking() {
  assert(is_tracking);
  is_tracking = false;
}

} // namespace hicma
