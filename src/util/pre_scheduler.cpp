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

struct kernel_args {
  void (*kernel)(
    float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
    const std::vector<std::vector<float>>& x,
    int64_t row_start, int64_t col_start
  ) = nullptr;
  const std::vector<std::vector<float>>& x;
  int64_t row_start, col_start;
};

void kernel_cpu_starpu_interface(void* buffers[], void* cl_args) {
  float* A = (float *)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t A_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  struct kernel_args* args = (kernel_args*)cl_args;
  args->kernel(
    A, A_dim0, A_dim1, A_stride, args->x, args->row_start, args->col_start
  );
}

struct starpu_codelet kernel_cl;

void make_kernel_codelet() {
  starpu_codelet_init(&kernel_cl);
  kernel_cl.cpu_funcs[0] = kernel_cpu_starpu_interface;
  kernel_cl.cpu_funcs_name[0] = "kernel_cpu_func";
  kernel_cl.name = "Kernel";
  kernel_cl.nbuffers = 1;
  kernel_cl.modes[0] = STARPU_W;
}

class Kernel_task : public Task {
 public:
  kernel_args args;
  Kernel_task(
    void (*kernel)(
      float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
      const std::vector<std::vector<float>>& x,
      int64_t row_start, int64_t col_start
    ),
    Dense& A, const std::vector<std::vector<float>>& x,
    int64_t row_start, int64_t col_start
  ) : Task({}, {A}), args{kernel, x, row_start, col_start} {
    if (schedule_started) {
      task = starpu_task_create();
      task->cl = &kernel_cl;
      task->cl_arg = &args;
      task->cl_arg_size = sizeof(args);
      task->handles[0] = get_handle(A);
    }
  }

  void submit() override {
    Dense& A = modified[0];
    if (schedule_started) {
      STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "kernel_task");
    } else {
      args.kernel(
        &A, A.dim[0], A.dim[1], A.stride, args.x, args.row_start, args.col_start
    );
    }
  }
};

void add_kernel_task(
  void (*kernel)(
    float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
    const std::vector<std::vector<float>>& x,
    int64_t row_start, int64_t col_start
  ),
  Dense& A, const std::vector<std::vector<float>>& x,
  int64_t row_start, int64_t col_start
) {
  add_task(std::make_shared<Kernel_task>(kernel, A, x, row_start, col_start));
}

struct copy_args { int64_t row_start, col_start; };

void copy_cpu_func(
  const float* A, uint64_t A_stride,
  float* B, uint64_t B_dim0, uint64_t B_dim1, uint64_t B_stride,
  copy_args& args
) {
  if (args.row_start == 0 && args.col_start == 0) {
    for (uint64_t i=0; i<B_dim0; i++) {
      for (uint64_t j=0; j<B_dim1; j++) {
        B[i*B_stride+j] = A[i*A_stride+j];
      }
    }
  } else {
    for (uint64_t i=0; i<B_dim0; i++) {
      for (uint64_t j=0; j<B_dim1; j++) {
        B[i*B_stride+j] = A[(args.row_start+i)*A_stride+args.col_start+j];
      }
    }
  }
}

void copy_cpu_starpu_interface(void* buffers[], void* cl_args) {
  const float* A = (float *)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  float* B = (float *)STARPU_MATRIX_GET_PTR(buffers[1]);
  uint64_t B_dim0 = STARPU_MATRIX_GET_NY(buffers[1]);
  uint64_t B_dim1 = STARPU_MATRIX_GET_NX(buffers[1]);
  uint64_t B_stride = STARPU_MATRIX_GET_LD(buffers[1]);
  struct copy_args* args = (copy_args*)cl_args;
  copy_cpu_func(A, A_stride, B, B_dim0, B_dim1, B_stride, *args);
}

struct starpu_codelet copy_cl;

void make_copy_codelet() {
  starpu_codelet_init(&copy_cl);
  copy_cl.cpu_funcs[0] = copy_cpu_starpu_interface;
  copy_cl.cpu_funcs_name[0] = "copy_cpu_func";
  copy_cl.name = "Copy";
  copy_cl.nbuffers = 2;
  copy_cl.modes[0] = STARPU_R;
  copy_cl.modes[1] = STARPU_W;
}

class Copy_task : public Task {
 public:
  copy_args args;
  Copy_task(const Dense& A, Dense& B, int64_t row_start=0, int64_t col_start=0)
  : Task({A}, {B}), args{row_start, col_start} {
    if (schedule_started) {
      task = starpu_task_create();
      task->cl = &copy_cl;
      task->cl_arg = &args;
      task->cl_arg_size = sizeof(args);
      task->handles[0] = get_handle(A);
      task->handles[1] = get_handle(B);
    }
  }

  void submit() override {
    const Dense& A = constant[0];
    Dense& B = modified[0];
    if (schedule_started) {
      STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "copy_task");
    } else {
      copy_cpu_func(&A, A.stride, &B, B.dim[0], B.dim[1], B.stride, args);
    }
  }
};

void add_copy_task(
  const Dense& A, Dense& B, int64_t row_start, int64_t col_start
) {
  add_task(std::make_shared<Copy_task>(A, B, row_start, col_start));
}

void transpose_cpu_func(
  const float* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  float* B, uint64_t B_stride
) {
  for (uint64_t i=0; i<A_dim0; i++) {
    for (uint64_t j=0; j<A_dim1; j++) {
      B[j*B_stride+i] = A[i*A_stride+j];
    }
  }
}

void transpose_cpu_starpu_interface(void* buffers[], void*) {
  const float* A = (float *)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t A_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  float* B = (float *)STARPU_MATRIX_GET_PTR(buffers[1]);
  uint64_t B_stride = STARPU_MATRIX_GET_LD(buffers[1]);
  transpose_cpu_func(A, A_dim0, A_dim1, A_stride, B, B_stride);
}

struct starpu_codelet transpose_cl;

void make_transpose_codelet() {
  starpu_codelet_init(&transpose_cl);
  transpose_cl.cpu_funcs[0] = transpose_cpu_starpu_interface;
  transpose_cl.cpu_funcs_name[0] = "transpose_cpu_func";
  transpose_cl.name = "Transpose";
  transpose_cl.nbuffers = 2;
  transpose_cl.modes[0] = STARPU_R;
  transpose_cl.modes[1] = STARPU_W;
}

class Transpose_task : public Task {
 public:
  Transpose_task(const Dense& A, Dense& B) : Task({A}, {B}) {
    if (schedule_started) {
      task = starpu_task_create();
      task->cl = &transpose_cl;
      task->handles[0] = get_handle(A);
      task->handles[1] = get_handle(B);
    }
  }

  void submit() override {
    const Dense& A = constant[0];
    Dense& B = modified[0];
    if (schedule_started) {
      STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "transpose_task");
    } else {
      transpose_cpu_func(&A, A.dim[0], A.dim[1], A.stride, &B, B.stride);
    }
  }
};

void add_transpose_task(const Dense& A, Dense& B) {
  add_task(std::make_shared<Transpose_task>(A, B));
}

struct assign_args { float value; };

void assign_cpu_func(
  float* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  assign_args& args
) {
  for (uint64_t i=0; i<A_dim0; i++) {
    for (uint64_t j=0; j<A_dim1; j++) {
      A[i*A_stride+j] = args.value;
    }
  }
}

void assign_cpu_starpu_interface(void* buffers[], void* cl_args) {
  float* A = (float *)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t A_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  struct assign_args* args = (assign_args*)cl_args;
  assign_cpu_func(A, A_dim0, A_dim1, A_stride, *args);
}

struct starpu_codelet assign_cl;

void make_assign_codelet() {
  starpu_codelet_init(&assign_cl);
  assign_cl.cpu_funcs[0] = assign_cpu_starpu_interface;
  assign_cl.cpu_funcs_name[0] = "assign_cpu_func";
  assign_cl.name = "Assign";
  assign_cl.nbuffers = 1;
  assign_cl.modes[0] = STARPU_W;
}

class Assign_task : public Task {
 public:
  assign_args args;
  Assign_task(Dense& A, float value) : Task({}, {A}), args{value} {
    if (schedule_started) {
      task = starpu_task_create();
      task->cl = &assign_cl;
      task->cl_arg = &args;
      task->cl_arg_size = sizeof(args);
      task->handles[0] = get_handle(A);
    }
  }

  void submit() override {
    Dense& A = modified[0];
    if (schedule_started) {
      STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "assign_task");
    } else {
      assign_cpu_func(&A, A.dim[0], A.dim[1], A.stride, args);
    }
  }
};

void add_assign_task(Dense& A, float value) {
  add_task(std::make_shared<Assign_task>(A, value));
}

void addition_cpu_func(
  float* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  const float* B, uint64_t B_stride
) {
  for (uint64_t i=0; i<A_dim0; i++) {
    for (uint64_t j=0; j<A_dim1; j++) {
      A[i*A_stride+j] += B[i*B_stride+j];
    }
  }
}

void addition_cpu_starpu_interface(void* buffers[], void*) {
  float* A = (float *)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t A_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  const float* B = (float *)STARPU_MATRIX_GET_PTR(buffers[1]);
  uint64_t B_stride = STARPU_MATRIX_GET_LD(buffers[1]);
  addition_cpu_func(A, A_dim0, A_dim1, A_stride, B, B_stride);
}

struct starpu_codelet addition_cl;

void make_addition_codelet() {
  starpu_codelet_init(&addition_cl);
  addition_cl.cpu_funcs[0] = addition_cpu_starpu_interface;
  addition_cl.cpu_funcs_name[0] = "addition_cpu_func";
  addition_cl.name = "Addition";
  addition_cl.nbuffers = 2;
  addition_cl.modes[0] = STARPU_RW;
  addition_cl.modes[1] = STARPU_R;
}

class Addition_task : public Task {
 public:
  Addition_task(Dense& A, const Dense& B) : Task({B}, {A}) {
    if (schedule_started) {
      task = starpu_task_create();
      task->cl = &addition_cl;
      task->handles[0] = get_handle(A);
      task->handles[1] = get_handle(B);
    }
  }

  void submit() override {
    Dense& A = modified[0];
    const Dense& B = constant[0];
    if (schedule_started) {
      STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "addition_task");
    } else {
      addition_cpu_func(&A, A.dim[0], A.dim[1], A.stride, &B, B.stride);
    }
  }
};

void add_addition_task(Dense& A, const Dense& B) {
  if (!matrix_is_tracked("addition_task", A, B) || !is_tracking) {
    add_task(std::make_shared<Addition_task>(A, B));
    register_matrix("addition_task", A, B);
  }
}

void subtraction_cpu_func(
  float* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  const float* B, uint64_t B_stride
) {
  for (uint64_t i=0; i<A_dim0; i++) {
    for (uint64_t j=0; j<A_dim1; j++) {
      A[i*A_stride+j] -= B[i*B_stride+j];
    }
  }
}

void subtraction_cpu_starpu_interface(void* buffers[], void*) {
  float* A = (float *)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t A_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  const float* B = (float *)STARPU_MATRIX_GET_PTR(buffers[1]);
  uint64_t B_stride = STARPU_MATRIX_GET_LD(buffers[1]);
  subtraction_cpu_func(A, A_dim0, A_dim1, A_stride, B, B_stride);
}

struct starpu_codelet subtraction_cl;

void make_subtraction_codelet() {
  starpu_codelet_init(&subtraction_cl);
  subtraction_cl.cpu_funcs[0] = subtraction_cpu_starpu_interface;
  subtraction_cl.cpu_funcs_name[0] = "subtraction_cpu_func";
  subtraction_cl.name = "Subtraction";
  subtraction_cl.nbuffers = 2;
  subtraction_cl.modes[0] = STARPU_RW;
  subtraction_cl.modes[1] = STARPU_R;
}

class Subtraction_task : public Task {
 public:
  Subtraction_task(Dense& A, const Dense& B) : Task({B}, {A}) {
    if (schedule_started) {
      task = starpu_task_create();
      task->cl = &subtraction_cl;
      task->handles[0] = get_handle(A);
      task->handles[1] = get_handle(B);
    }
  }

  void submit() override {
    Dense& A = modified[0];
    const Dense& B = constant[0];
    if (schedule_started) {
      STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "subtraction_task");
    } else {
      subtraction_cpu_func(&A, A.dim[0], A.dim[1], A.stride, &B, B.stride);
    }
  }
};

void add_subtraction_task(Dense& A, const Dense& B) {
  add_task(std::make_shared<Subtraction_task>(A, B));
}

struct multiplication_args { float factor; };

void multiplication_cpu_func(
  float* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  multiplication_args& args
) {
  for (uint64_t i=0; i<A_dim0; i++) {
    for (uint64_t j=0; j<A_dim1; j++) {
      A[i*A_stride+j] *= args.factor;
    }
  }
}

void multiplication_cpu_starpu_interface(void* buffers[], void* cl_args) {
  float* A = (float *)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t A_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  struct multiplication_args* args = (multiplication_args*)cl_args;
  multiplication_cpu_func(A, A_dim0, A_dim1, A_stride, *args);
}

struct starpu_codelet multiplication_cl;

void make_multiplication_codelet() {
  starpu_codelet_init(&multiplication_cl);
  multiplication_cl.cpu_funcs[0] = multiplication_cpu_starpu_interface;
  multiplication_cl.cpu_funcs_name[0] = "multiplication_cpu_func";
  multiplication_cl.name = "Multiplication";
  multiplication_cl.nbuffers = 1;
  multiplication_cl.modes[0] = STARPU_W;
}

class Multiplication_task : public Task {
 public:
  multiplication_args args;
  Multiplication_task(Dense& A, float factor) : Task({}, {A}), args{factor} {
    if (schedule_started) {
      task = starpu_task_create();
      task->cl = &multiplication_cl;
      task->cl_arg = &args;
      task->cl_arg_size = sizeof(args);
      task->handles[0] = get_handle(A);
    }
  }

  void submit() override {
    Dense& A = modified[0];
    if (schedule_started) {
      STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "multiplication_task");
    } else {
      multiplication_cpu_func(&A, A.dim[0], A.dim[1], A.stride, args);
    }
  }
};

void add_multiplication_task(Dense& A, float factor) {
  // Don't do anything if factor == 1
  if (factor != 1) {
    add_task(std::make_shared<Multiplication_task>(A, factor));
  }
}

void getrf_cpu_func(
  float* AU, uint64_t AU_dim0, uint64_t AU_dim1, uint64_t AU_stride,
  float* L, uint64_t L_stride
) {
  std::vector<int> ipiv(std::min(AU_dim0, AU_dim1));
  LAPACKE_sgetrf(
    LAPACK_ROW_MAJOR,
    AU_dim0, AU_dim1,
    AU, AU_stride,
    &ipiv[0]
  );
  for (uint64_t i=0; i<AU_dim0; i++) {
    for (uint64_t j=0; j<i; j++) {
      L[i*L_stride+j] = AU[i*AU_stride+j];
      AU[i*AU_stride+j] = 0;
    }
    L[i*L_stride+i] = 1;
  }
}

void getrf_cpu_starpu_interface(void* buffers[], void*) {
  float* AU = (float*)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t AU_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t AU_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t AU_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  float* L = (float*)STARPU_MATRIX_GET_PTR(buffers[1]);
  uint64_t L_stride = STARPU_MATRIX_GET_LD(buffers[1]);
  getrf_cpu_func(AU, AU_dim0, AU_dim1, AU_stride, L, L_stride);
}

struct starpu_codelet getrf_cl;

void make_getrf_codelet() {
  starpu_codelet_init(&getrf_cl);
  getrf_cl.cpu_funcs[0] = getrf_cpu_starpu_interface;
  getrf_cl.cpu_funcs_name[0] = "getrf_cpu_func";
  getrf_cl.name = "GETRF";
  getrf_cl.nbuffers = 2;
  getrf_cl.modes[0] = STARPU_RW;
  getrf_cl.modes[1] = STARPU_W;
}

class GETRF_task : public Task {
 public:
  GETRF_task(Dense& AU, Dense& L) : Task({}, {AU, L}) {
    if (schedule_started) {
      task = starpu_task_create();
      task->cl = &getrf_cl;
      task->handles[0] = get_handle(AU);
      task->handles[1] = get_handle(L);
    }
  }

  void submit() override {
    Dense& AU = modified[0];
    Dense& L = modified[1];
    if (schedule_started) {
      STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "getrf_task");
    } else {
      getrf_cpu_func(&AU, AU.dim[0], AU.dim[1], AU.stride, &L, L.stride);
    }
  }
};

void add_getrf_task(Dense& AU, Dense& L) {
  add_task(std::make_shared<GETRF_task>(AU, L));
}

void qr_cpu_func(
  float* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  float* Q, uint64_t Q_dim0, uint64_t Q_dim1, uint64_t Q_stride,
  float* R, uint64_t, uint64_t, uint64_t R_stride
) {
  uint64_t k = std::min(A_dim0, A_dim1);
  std::vector<float> tau(k);
  for (uint64_t i=0; i<std::min(Q_dim0, Q_dim1); i++) Q[i*Q_stride+i] = 1.0;
  LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, A_dim0, A_dim1, A, A_stride, &tau[0]);
  // TODO Consider using A for the dorgqr and moving to Q afterwards! That
  // also simplify this loop.
  for(uint64_t i=0; i<Q_dim0; i++) {
    for(uint64_t j=i; j<Q_dim1; j++) {
      R[i*R_stride+j] = A[i*A_stride+j];
    }
  }
  for(uint64_t i=0; i<Q_dim0; i++) {
    for(uint64_t j=0; j<std::min(i, Q_dim1); j++) {
      Q[i*Q_stride+j] = A[i*A_stride+j];
    }
  }
  // TODO Consider making special function for this. Performance heavy
  // and not always needed. If Q should be applied to something, use directly!
  // Alternatively, create Dense deriative that remains in elementary
  // reflector form, uses dormqr instead of gemm and can be transformed to
  // Dense via dorgqr!
  LAPACKE_sorgqr(
    LAPACK_ROW_MAJOR, Q_dim0, Q_dim1, k, Q, Q_stride, &tau[0]
  );
}

void qr_cpu_starpu_interface(void* buffers[], void*) {
  float* A = (float*)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t A_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  float* Q = (float*)STARPU_MATRIX_GET_PTR(buffers[1]);
  uint64_t Q_dim0 = STARPU_MATRIX_GET_NY(buffers[1]);
  uint64_t Q_dim1 = STARPU_MATRIX_GET_NX(buffers[1]);
  uint64_t Q_stride = STARPU_MATRIX_GET_LD(buffers[1]);
  float* R = (float*)STARPU_MATRIX_GET_PTR(buffers[2]);
  uint64_t R_dim0 = STARPU_MATRIX_GET_NY(buffers[2]);
  uint64_t R_dim1 = STARPU_MATRIX_GET_NX(buffers[2]);
  uint64_t R_stride = STARPU_MATRIX_GET_LD(buffers[2]);
  qr_cpu_func(
    A, A_dim0, A_dim1, A_stride,
    Q, Q_dim0, Q_dim1, Q_stride,
    R, R_dim0, R_dim1, R_stride
  );
}

struct starpu_codelet qr_cl;

void make_qr_codelet() {
  starpu_codelet_init(&qr_cl);
  qr_cl.cpu_funcs[0] = qr_cpu_starpu_interface;
  qr_cl.cpu_funcs_name[0] = "qr_cpu_func";
  qr_cl.name = "QR";
  qr_cl.nbuffers = 3;
  qr_cl.modes[0] = STARPU_RW;
  qr_cl.modes[1] = STARPU_W;
  qr_cl.modes[2] = STARPU_W;
}

class QR_task : public Task {
 public:
  QR_task(Dense& A, Dense& Q, Dense& R) : Task({}, {A, Q, R}) {
    if (schedule_started) {
      task = starpu_task_create();
      task->cl = &qr_cl;
      task->handles[0] = get_handle(A);
      task->handles[1] = get_handle(Q);
      task->handles[2] = get_handle(R);
    }
  }

  void submit() override {
    Dense& A = modified[0];
    Dense& Q = modified[1];
    Dense& R = modified[2];
    if (schedule_started) {
      STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "qr_task");
    } else {
      qr_cpu_func(
        &A, A.dim[0], A.dim[1], A.stride,
        &Q, Q.dim[0], Q.dim[1], Q.stride,
        &R, R.dim[0], R.dim[1], R.stride
      );
    }
  }
};

void add_qr_task(Dense& A, Dense& Q, Dense& R) {
  // TODO Check for duplicate/shared tasks
  add_task(std::make_shared<QR_task>(A, Q, R));
}

void rq_cpu_func(
  float* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  float* R, uint64_t R_dim0, uint64_t R_dim1, uint64_t R_stride,
  float* Q, uint64_t Q_dim0, uint64_t Q_dim1, uint64_t Q_stride
) {
  uint64_t k = std::min(A_dim0, A_dim1);
  std::vector<float> tau(k);
  LAPACKE_sgerqf(LAPACK_ROW_MAJOR, A_dim0, A_dim1, A, A_stride, &tau[0]);
  // TODO Consider making special function for this. Performance heavy and not
  // always needed. If Q should be applied to something, use directly!
  // Alternatively, create Dense deriative that remains in elementary reflector
  // form, uses dormqr instead of gemm and can be transformed to Dense via
  // dorgqr!
  for (uint64_t i=0; i<R_dim0; i++) {
    for (uint64_t j=i; j<R_dim1; j++) {
      R[i*R_stride+j] = A[i*A_stride+A_dim1-R_dim1+j];
    }
  }
  for(uint64_t i=0; i<Q_dim0; i++) {
    for(uint64_t j=0; j<std::min(A_dim1-R_dim1+i, Q_dim1); j++) {
      Q[i*Q_stride+j] = A[i*A_stride+j];
    }
  }
  LAPACKE_sorgrq(
    LAPACK_ROW_MAJOR, Q_dim0, Q_dim1, k, Q, Q_stride, &tau[0]
  );
}

void rq_cpu_starpu_interface(void* buffers[], void*) {
  float* A = (float*)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t A_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  float* R = (float*)STARPU_MATRIX_GET_PTR(buffers[1]);
  uint64_t R_dim0 = STARPU_MATRIX_GET_NY(buffers[1]);
  uint64_t R_dim1 = STARPU_MATRIX_GET_NX(buffers[1]);
  uint64_t R_stride = STARPU_MATRIX_GET_LD(buffers[1]);
  float* Q = (float*)STARPU_MATRIX_GET_PTR(buffers[2]);
  uint64_t Q_dim0 = STARPU_MATRIX_GET_NY(buffers[2]);
  uint64_t Q_dim1 = STARPU_MATRIX_GET_NX(buffers[2]);
  uint64_t Q_stride = STARPU_MATRIX_GET_LD(buffers[2]);
  rq_cpu_func(
    A, A_dim0, A_dim1, A_stride,
    R, R_dim0, R_dim1, R_stride,
    Q, Q_dim0, Q_dim1, Q_stride
  );
}

struct starpu_codelet rq_cl;

void make_rq_codelet() {
  starpu_codelet_init(&rq_cl);
  rq_cl.cpu_funcs[0] = rq_cpu_starpu_interface;
  rq_cl.cpu_funcs_name[0] = "rq_cpu_func";
  rq_cl.name = "RQ";
  rq_cl.nbuffers = 3;
  rq_cl.modes[0] = STARPU_RW;
  rq_cl.modes[1] = STARPU_W;
  rq_cl.modes[2] = STARPU_W;
}

class RQ_task : public Task {
 public:
  RQ_task(Dense& A, Dense& R, Dense& Q) : Task({}, {A, R, Q}) {
    if (schedule_started) {
      task = starpu_task_create();
      task->cl = &rq_cl;
      task->handles[0] = get_handle(A);
      task->handles[1] = get_handle(R);
      task->handles[2] = get_handle(Q);
    }
  }

  void submit() override {
    Dense& A = modified[0];
    Dense& R = modified[1];
    Dense& Q = modified[2];
    if (schedule_started) {
      STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "rq_task");
    } else {
      rq_cpu_func(
        &A, A.dim[0], A.dim[1], A.stride,
        &R, R.dim[0], R.dim[1], R.stride,
        &Q, Q.dim[0], Q.dim[1], Q.stride
      );
    }
  }
};

void add_rq_task(Dense& A, Dense& R, Dense& Q) {
  // TODO Check for duplicate/shared tasks
  add_task(std::make_shared<RQ_task>(A, R, Q));
}

struct trsm_args { int uplo; int lr; };

void trsm_cpu_func(
  const float* A, uint64_t A_stride,
  float* B, uint64_t B_dim0, uint64_t B_dim1, uint64_t B_stride,
  trsm_args& args
) {
  cblas_strsm(
    CblasRowMajor,
    args.lr==TRSM_LEFT?CblasLeft:CblasRight,
    args.uplo==TRSM_UPPER?CblasUpper:CblasLower,
    CblasNoTrans,
    args.uplo==TRSM_UPPER?CblasNonUnit:CblasUnit,
    B_dim0, B_dim1,
    1,
    A, A_stride,
    B, B_stride
  );
}

void trsm_cpu_starpu_interface(void* buffers[], void* cl_args) {
  const float* A = (float *)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  float* B = (float *)STARPU_MATRIX_GET_PTR(buffers[1]);
  uint64_t B_dim0 = STARPU_MATRIX_GET_NY(buffers[1]);
  uint64_t B_dim1 = STARPU_MATRIX_GET_NX(buffers[1]);
  uint64_t B_stride = STARPU_MATRIX_GET_LD(buffers[1]);
  struct trsm_args* args = (trsm_args*)cl_args;
  trsm_cpu_func(A, A_stride, B, B_dim0, B_dim1, B_stride, *args);
}

struct starpu_codelet trsm_cl;

void make_trsm_codelet() {
  starpu_codelet_init(&trsm_cl);
  trsm_cl.cpu_funcs[0] = trsm_cpu_starpu_interface;
  trsm_cl.cpu_funcs_name[0] = "trsm_cpu_func";
  trsm_cl.name = "TRSM";
  trsm_cl.nbuffers = 2;
  trsm_cl.modes[0] = STARPU_R;
  trsm_cl.modes[1] = STARPU_RW;
}

class TRSM_task : public Task {
 public:
  trsm_args args;
  TRSM_task(const Dense& A, Dense& B, int uplo, int lr)
  : Task({A}, {B}), args{uplo, lr} {
    if (schedule_started) {
      task = starpu_task_create();
      task->cl = &trsm_cl;
      task->cl_arg = &args;
      task->cl_arg_size = sizeof(args);
      task->handles[0] = get_handle(A);
      task->handles[1] = get_handle(B);
    }
  }

  void submit() override {
    const Dense& A = constant[0];
    Dense& B = modified[0];
    if (schedule_started) {
      STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "trsm_task");
    } else {
      trsm_cpu_func(&A, A.stride, &B, B.dim[0], B.dim[1], B.stride, args);
    }
  }
};

void add_trsm_task(const Dense& A, Dense& B, int uplo, int lr) {
  if (!matrix_is_tracked("trsm_task", A, B) || !is_tracking) {
    add_task(std::make_shared<TRSM_task>(A, B, uplo, lr));
    register_matrix("trsm_task", A, B);
  }
}

struct gemm_args { float alpha, beta; bool TransA, TransB;  };

void gemm_cpu_func(
  const float* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  const float* B, uint64_t, uint64_t B_dim1, uint64_t B_stride,
  float* C, uint64_t C_dim0, uint64_t C_dim1, uint64_t C_stride,
  gemm_args& args
) {
  if (B_dim1 == 1) {
    cblas_sgemv(
      CblasRowMajor,
      CblasNoTrans,
      A_dim0, A_dim1,
      args.alpha,
      A, A_stride,
      B, B_stride,
      args.beta,
      C, C_stride
    );
  }
  else {
    int64_t k = args.TransA ? A_dim0 : A_dim1;
    cblas_sgemm(
      CblasRowMajor,
      args.TransA?CblasTrans:CblasNoTrans, args.TransB?CblasTrans:CblasNoTrans,
      C_dim0, C_dim1, k,
      args.alpha,
      A, A_stride,
      B, B_stride,
      args.beta,
      C, C_stride
    );
  }
}

void gemm_cpu_starpu_interface(void* buffers[], void* cl_args) {
  const float* A = (float *)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t A_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  const float* B = (float *)STARPU_MATRIX_GET_PTR(buffers[1]);
  uint64_t B_dim0 = STARPU_MATRIX_GET_NY(buffers[1]);
  uint64_t B_dim1 = STARPU_MATRIX_GET_NX(buffers[1]);
  uint64_t B_stride = STARPU_MATRIX_GET_LD(buffers[1]);
  float* C = (float *)STARPU_MATRIX_GET_PTR(buffers[2]);
  uint64_t C_dim0 = STARPU_MATRIX_GET_NY(buffers[2]);
  uint64_t C_dim1 = STARPU_MATRIX_GET_NX(buffers[2]);
  uint64_t C_stride = STARPU_MATRIX_GET_LD(buffers[2]);
  struct gemm_args* args = (gemm_args*)cl_args;
  gemm_cpu_func(
    A, A_dim0, A_dim1, A_stride,
    B, B_dim0, B_dim1, B_stride,
    C, C_dim0, C_dim1, C_stride,
    *args
  );
}

struct starpu_codelet gemm_cl;

void make_gemm_codelet() {
  starpu_codelet_init(&gemm_cl);
  gemm_cl.cpu_funcs[0] = gemm_cpu_starpu_interface;
  gemm_cl.cpu_funcs_name[0] = "gemm_cpu_func";
  gemm_cl.name = "GEMM";
  gemm_cl.nbuffers = 3;
  gemm_cl.modes[0] = STARPU_R;
  gemm_cl.modes[1] = STARPU_R;
  gemm_cl.modes[2] = STARPU_RW;
}

class GEMM_task : public Task {
 public:
  gemm_args args;
  GEMM_task(
    const Dense& A, const Dense& B, Dense& C,
    float alpha, float beta, bool TransA, bool TransB
  ) : Task({A, B}, {C}), args{alpha, beta, TransA, TransB} {
    if (schedule_started) {
      task = starpu_task_create();
      task->cl = &gemm_cl;
      task->cl_arg = &args;
      task->cl_arg_size = sizeof(args);
      task->handles[0] = get_handle(A);
      task->handles[1] = get_handle(B);
      task->handles[2] = get_handle(C);
      // Effectively write only, this might be important for dependencies
      if (args.beta == 0) STARPU_TASK_SET_MODE(task, STARPU_W, 2);
    }
  }

  void submit() override {
    const Dense& A = constant[0];
    const Dense& B = constant[1];
    Dense& C = modified[0];
    if (schedule_started) {
      STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "gemm_task");
    } else {
      gemm_cpu_func(
        &A, A.dim[0], A.dim[1], A.stride,
        &B, B.dim[0], B.dim[1], B.stride,
        &C, C.dim[0], C.dim[1], C.stride,
        args
      );
    }
  }
};

BasisTracker<
  BasisKey, BasisTracker<BasisKey, std::shared_ptr<GEMM_task>>
> gemm_tracker;

void add_gemm_task(
  const Dense& A, const Dense& B, Dense& C,
  float alpha, float beta, bool TransA, bool TransB
) {
  if (
    is_tracking
    && !C.is_submatrix()
    && gemm_tracker.has_key(A) && gemm_tracker[A].has_key(B)
    && gemm_tracker[A][B]->args.alpha == alpha
    && gemm_tracker[A][B]->args.beta == beta
    && gemm_tracker[A][B]->args.TransA == TransA
    && gemm_tracker[A][B]->args.TransB == TransB
  ) {
    if (C.is_shared_with(gemm_tracker[A][B]->modified[0])) {
      return;
    } else
    if (beta == 0) {
      C = gemm_tracker[A][B]->modified[0].shallow_copy();
      return;
    }
  }
  std::shared_ptr<GEMM_task> task = std::make_shared<GEMM_task>(
    A, B, C, alpha, beta, TransA, TransB
  );
  if (is_tracking && !C.is_submatrix()) {
    gemm_tracker[A][B] = task;
  }
  add_task(task);
}

void svd_cpu_func(
  float* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  float* U, uint64_t, uint64_t, uint64_t U_stride,
  float* S, uint64_t S_dim0, uint64_t, uint64_t S_stride,
  float* V, uint64_t, uint64_t, uint64_t V_stride
) {
  std::vector<float> Sdiag(S_dim0, 0);
  std::vector<float> work(S_dim0-1, 0);
  LAPACKE_sgesvd(
    LAPACK_ROW_MAJOR,
    'S', 'S',
    A_dim0, A_dim1,
    A, A_stride,
    &Sdiag[0],
    U, U_stride,
    V, V_stride,
    &work[0]
  );
  for(uint64_t i=0; i<S_dim0; i++){
    S[i*S_stride+i] = Sdiag[i];
  }
}

void svd_cpu_starpu_interface(void* buffers[], void*) {
  float* A = (float *)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t A_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  float* U = (float *)STARPU_MATRIX_GET_PTR(buffers[1]);
  uint64_t U_dim0 = STARPU_MATRIX_GET_NY(buffers[1]);
  uint64_t U_dim1 = STARPU_MATRIX_GET_NX(buffers[1]);
  uint64_t U_stride = STARPU_MATRIX_GET_LD(buffers[1]);
  float* S = (float *)STARPU_MATRIX_GET_PTR(buffers[2]);
  uint64_t S_dim0 = STARPU_MATRIX_GET_NY(buffers[2]);
  uint64_t S_dim1 = STARPU_MATRIX_GET_NX(buffers[2]);
  uint64_t S_stride = STARPU_MATRIX_GET_LD(buffers[2]);
  float* V = (float *)STARPU_MATRIX_GET_PTR(buffers[3]);
  uint64_t V_dim0 = STARPU_MATRIX_GET_NY(buffers[3]);
  uint64_t V_dim1 = STARPU_MATRIX_GET_NX(buffers[3]);
  uint64_t V_stride = STARPU_MATRIX_GET_LD(buffers[3]);
  svd_cpu_func(
    A, A_dim0, A_dim1, A_stride,
    U, U_dim0, U_dim1, U_stride,
    S, S_dim0, S_dim1, S_stride,
    V, V_dim0, V_dim1, V_stride
  );
}

struct starpu_codelet svd_cl;

void make_svd_codelet() {
  starpu_codelet_init(&svd_cl);
  svd_cl.cpu_funcs[0] = svd_cpu_starpu_interface;
  svd_cl.cpu_funcs_name[0] = "svd_cpu_func";
  svd_cl.name = "SVD";
  svd_cl.nbuffers = 4;
  svd_cl.modes[0] = STARPU_RW;
  svd_cl.modes[1] = STARPU_W;
  svd_cl.modes[2] = STARPU_W;
  svd_cl.modes[3] = STARPU_W;
}

class SVD_task : public Task {
 public:
  SVD_task(Dense& A, Dense& U, Dense& S, Dense& V) : Task({}, {A, U, S, V}) {
    if (schedule_started) {
      task = starpu_task_create();
      task->cl = &svd_cl;
      task->handles[0] = get_handle(A);
      task->handles[1] = get_handle(U);
      task->handles[2] = get_handle(S);
      task->handles[3] = get_handle(V);
    }
  }

  void submit() override {
    Dense& A = modified[0];
    Dense& U = modified[1];
    Dense& S = modified[2];
    Dense& V = modified[3];
    if (schedule_started) {
      STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "svd_task");
    } else {
      svd_cpu_func(
        &A, A.dim[0], A.dim[1], A.stride,
        &U, U.dim[0], U.dim[1], U.stride,
        &S, S.dim[0], S.dim[1], S.stride,
        &V, V.dim[0], V.dim[1], V.stride
      );
    }
  }
};

void add_svd_task(Dense& A, Dense& U, Dense& S, Dense& V) {
  add_task(std::make_shared<SVD_task>(A, U, S, V));
}

void start_schedule() {
  assert(!schedule_started);
  assert(tasks.empty());
  schedule_started = true;
  clear_task_trackers();
}

void execute_schedule() {
  for (std::shared_ptr<Task> task : tasks) {
    task->submit();
  }
  starpu_task_wait_for_all();
  tasks.clear();
  clear_task_trackers();
  schedule_started = false;
}

void initialize_starpu() {
  STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "init");
  make_kernel_codelet();
  make_copy_codelet();
  make_transpose_codelet();
  make_assign_codelet();
  make_addition_codelet();
  make_subtraction_codelet();
  make_multiplication_codelet();
  make_getrf_codelet();
  make_qr_codelet();
  make_rq_codelet();
  make_trsm_codelet();
  make_gemm_codelet();
  make_svd_codelet();
}

void clear_task_trackers() {
  gemm_tracker.clear();
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
