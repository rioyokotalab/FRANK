#include "hicma/util/pre_scheduler.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/classes/initialization_helpers/basis_tracker.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"

#include "starpu.h"
#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <utility>
#include <vector>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif


namespace hicma
{

Task::Task(
  std::vector<std::reference_wrapper<const Dense>> constant_,
  std::vector<std::reference_wrapper<Dense>> modified_
) {
  for (size_t i=0; i<constant_.size(); ++i) {
    constant.push_back(constant_[i].get().share());
  }
  for (size_t i=0; i<modified_.size(); ++i) {
    modified.push_back(modified_[i].get().share());
  }
}

std::list<std::shared_ptr<Task>> tasks;
bool schedule_started = false;

void add_task(std::shared_ptr<Task> task) {
  if (schedule_started) {
    tasks.push_back(task);
  } else {
    task->execute();
  }
}

Kernel_task::Kernel_task(
  void (*kernel)(
    double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
    const std::vector<std::vector<double>>& x,
    int64_t row_start, int64_t col_start
  ),
  Dense& A, const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
) : Task({}, {A}), args{kernel, x, row_start, col_start} {}

void kernel_cpu_starpu_interface(void* buffers[], void* cl_args) {
  double* A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
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

void Kernel_task::execute() {
  Dense& A = modified[0];
  if (schedule_started) {
    struct starpu_task* task = starpu_task_create();
    task->cl = &kernel_cl;
    task->cl_arg = &args;
    task->cl_arg_size = sizeof(args);
    task->handles[0] = starpu_data_lookup(&A);
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "kernel_task");
  } else {
    args.kernel(
      &A, A.dim[0], A.dim[1], A.stride, args.x, args.row_start, args.col_start
  );
  }
}

Copy_task::Copy_task(
  const Dense& A, Dense& B, int64_t row_start, int64_t col_start
) : Task({A}, {B}), args{row_start, col_start} {}

void copy_cpu_func(
  const double* A, uint64_t A_stride,
  double* B, uint64_t B_dim0, uint64_t B_dim1, uint64_t B_stride,
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
  const double* A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  double* B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
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

void Copy_task::execute() {
  const Dense& A = constant[0];
  Dense& B = modified[0];
  if (schedule_started) {
    struct starpu_task* task = starpu_task_create();
    task->cl = &copy_cl;
    task->cl_arg = &args;
    task->cl_arg_size = sizeof(args);
    task->handles[0] = starpu_data_lookup(&A);
    task->handles[1] = starpu_data_lookup(&B);
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "copy_task");
  } else {
    copy_cpu_func(&A, A.stride, &B, B.dim[0], B.dim[1], B.stride, args);
  }
}

Assign_task::Assign_task(Dense& A, double value)
: Task({}, {A}), args{value} {}

void assign_cpu_func(
  double* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  assign_args& args
) {
  for (uint64_t i=0; i<A_dim0; i++) {
    for (uint64_t j=0; j<A_dim1; j++) {
      A[i*A_stride+j] = args.value;
    }
  }
}

void assign_cpu_starpu_interface(void* buffers[], void* cl_args) {
  double* A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
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

void Assign_task::execute() {
  Dense& A = modified[0];
  if (schedule_started) {
    struct starpu_task* task = starpu_task_create();
    task->cl = &assign_cl;
    task->cl_arg = &args;
    task->cl_arg_size = sizeof(args);
    task->handles[0] = starpu_data_lookup(&A);
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "assign_task");
  } else {
    assign_cpu_func(&A, A.dim[0], A.dim[1], A.stride, args);
  }
}

Addition_task::Addition_task(Dense& A, const Dense& B)
: Task({B}, {A}) {}

void addition_cpu_func(
  double* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  const double* B, uint64_t B_stride
) {
  for (uint64_t i=0; i<A_dim0; i++) {
    for (uint64_t j=0; j<A_dim1; j++) {
      A[i*A_stride+j] += B[i*B_stride+j];
    }
  }
}

void addition_cpu_starpu_interface(void* buffers[], void*) {
  double* A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t A_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  const double* B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
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

void Addition_task::execute() {
  Dense& A = modified[0];
  const Dense& B = constant[0];
  if (schedule_started) {
    struct starpu_task* task = starpu_task_create();
    task->cl = &addition_cl;
    task->handles[0] = starpu_data_lookup(&A);
    task->handles[1] = starpu_data_lookup(&B);
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "addition_task");
  } else {
    addition_cpu_func(&A, A.dim[0], A.dim[1], A.stride, &B, B.stride);
  }
}

Subtraction_task::Subtraction_task(Dense& A, const Dense& B)
: Task({B}, {A}) {}

void subtraction_cpu_func(
  double* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  const double* B, uint64_t B_stride
) {
  for (uint64_t i=0; i<A_dim0; i++) {
    for (uint64_t j=0; j<A_dim1; j++) {
      A[i*A_stride+j] -= B[i*B_stride+j];
    }
  }
}

void subtraction_cpu_starpu_interface(void* buffers[], void*) {
  double* A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t A_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  const double* B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
  uint64_t B_stride = STARPU_MATRIX_GET_LD(buffers[1]);
  subtraction_cpu_func(A, A_dim0, A_dim1, A_stride, B, B_stride);
}

struct starpu_codelet subtraction_cl;

void make_subtraction_codelet() {
  starpu_codelet_init(&subtraction_cl);
  subtraction_cl.cpu_funcs[0] = subtraction_cpu_starpu_interface;
  subtraction_cl.cpu_funcs_name[0] = "subtraction_cpu_func";
  subtraction_cl.name = "Subtraction";
  subtraction_cl.nbuffers = 1;
  subtraction_cl.modes[0] = STARPU_RW;
  subtraction_cl.modes[1] = STARPU_R;
}

void Subtraction_task::execute() {
  Dense& A = modified[0];
  const Dense& B = constant[0];
  if (schedule_started) {
    struct starpu_task* task = starpu_task_create();
    task->cl = &subtraction_cl;
    task->handles[0] = starpu_data_lookup(&A);
    task->handles[1] = starpu_data_lookup(&B);
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "subtraction_task");
  } else {
    subtraction_cpu_func(&A, A.dim[0], A.dim[1], A.stride, &B, B.stride);
  }
}

Multiplication_task::Multiplication_task(Dense& A, double factor)
: Task({}, {A}), args{factor} {}

void multiplication_cpu_func(
  double* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  multiplication_args& args
) {
  for (uint64_t i=0; i<A_dim0; i++) {
    for (uint64_t j=0; j<A_dim1; j++) {
      A[i*A_stride+j] *= args.factor;
    }
  }
}

void multiplication_cpu_starpu_interface(void* buffers[], void* cl_args) {
  double* A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
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

void Multiplication_task::execute() {
  Dense& A = modified[0];
  if (schedule_started) {
    struct starpu_task* task = starpu_task_create();
    task->cl = &multiplication_cl;
    task->cl_arg = &args;
    task->cl_arg_size = sizeof(args);
    task->handles[0] = starpu_data_lookup(&A);
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "multiplication_task");
  } else {
    multiplication_cpu_func(&A, A.dim[0], A.dim[1], A.stride, args);
  }
}

GETRF_task::GETRF_task(Dense& AU, Dense& L) : Task({}, {AU, L}) {}

void getrf_cpu_func(
  double* AU, uint64_t AU_dim0, uint64_t AU_dim1, uint64_t AU_stride,
  double* L, uint64_t L_stride
) {
  std::vector<int> ipiv(std::min(AU_dim0, AU_dim1));
  LAPACKE_dgetrf(
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
  double* AU = (double*)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t AU_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t AU_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t AU_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  double* L = (double*)STARPU_MATRIX_GET_PTR(buffers[1]);
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

void GETRF_task::execute() {
  Dense& AU = modified[0];
  Dense& L = modified[1];
  if (schedule_started) {
    struct starpu_task* task = starpu_task_create();
    task->cl = &getrf_cl;
    task->handles[0] = starpu_data_lookup(&AU);
    task->handles[1] = starpu_data_lookup(&L);
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "getrf_task");
  } else {
    getrf_cpu_func(&AU, AU.dim[0], AU.dim[1], AU.stride, &L, L.stride);
  }
}

QR_task::QR_task(Dense& A, Dense& Q, Dense& R) : Task({}, {A, Q, R}) {}

void qr_cpu_func(
  double* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  double* Q, uint64_t Q_dim0, uint64_t Q_dim1, uint64_t Q_stride,
  double* R, uint64_t, uint64_t, uint64_t R_stride
) {
  uint64_t k = std::min(A_dim0, A_dim1);
  std::vector<double> tau(k);
  for (uint64_t i=0; i<std::min(Q_dim0, Q_dim1); i++) Q[i*Q_stride+i] = 1.0;
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A_dim0, A_dim1, A, A_stride, &tau[0]);
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
  LAPACKE_dorgqr(
    LAPACK_ROW_MAJOR, Q_dim0, Q_dim1, k, Q, Q_stride, &tau[0]
  );
}

void qr_cpu_starpu_interface(void* buffers[], void*) {
  double* A = (double*)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t A_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  double* Q = (double*)STARPU_MATRIX_GET_PTR(buffers[1]);
  uint64_t Q_dim0 = STARPU_MATRIX_GET_NY(buffers[1]);
  uint64_t Q_dim1 = STARPU_MATRIX_GET_NX(buffers[1]);
  uint64_t Q_stride = STARPU_MATRIX_GET_LD(buffers[1]);
  double* R = (double*)STARPU_MATRIX_GET_PTR(buffers[2]);
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

void QR_task::execute() {
  Dense& A = modified[0];
  Dense& Q = modified[1];
  Dense& R = modified[2];
  if (schedule_started) {
    struct starpu_task* task = starpu_task_create();
    task->cl = &qr_cl;
    task->handles[0] = starpu_data_lookup(&A);
    task->handles[1] = starpu_data_lookup(&Q);
    task->handles[2] = starpu_data_lookup(&R);
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "qr_task");
  } else {
    qr_cpu_func(
      &A, A.dim[0], A.dim[1], A.stride,
      &Q, Q.dim[0], Q.dim[1], Q.stride,
      &R, R.dim[0], R.dim[1], R.stride
    );
  }
}

RQ_task::RQ_task(Dense& A, Dense& R, Dense& Q) : Task({}, {A, R, Q}) {}

void rq_cpu_func(
  double* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  double* R, uint64_t R_dim0, uint64_t R_dim1, uint64_t R_stride,
  double* Q, uint64_t Q_dim0, uint64_t Q_dim1, uint64_t Q_stride
) {
  uint64_t k = std::min(A_dim0, A_dim1);
  std::vector<double> tau(k);
  LAPACKE_dgerqf(LAPACK_ROW_MAJOR, A_dim0, A_dim1, A, A_stride, &tau[0]);
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
  LAPACKE_dorgrq(
    LAPACK_ROW_MAJOR, Q_dim0, Q_dim1, k, Q, Q_stride, &tau[0]
  );
}

void rq_cpu_starpu_interface(void* buffers[], void*) {
  double* A = (double*)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t A_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  double* R = (double*)STARPU_MATRIX_GET_PTR(buffers[1]);
  uint64_t R_dim0 = STARPU_MATRIX_GET_NY(buffers[1]);
  uint64_t R_dim1 = STARPU_MATRIX_GET_NX(buffers[1]);
  uint64_t R_stride = STARPU_MATRIX_GET_LD(buffers[1]);
  double* Q = (double*)STARPU_MATRIX_GET_PTR(buffers[2]);
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

void RQ_task::execute() {
  Dense& A = modified[0];
  Dense& R = modified[1];
  Dense& Q = modified[2];
  if (schedule_started) {
    struct starpu_task* task = starpu_task_create();
    task->cl = &rq_cl;
    task->handles[0] = starpu_data_lookup(&A);
    task->handles[1] = starpu_data_lookup(&R);
    task->handles[2] = starpu_data_lookup(&Q);
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "rq_task");
  } else {
    rq_cpu_func(
      &A, A.dim[0], A.dim[1], A.stride,
      &R, R.dim[0], R.dim[1], R.stride,
      &Q, Q.dim[0], Q.dim[1], Q.stride
    );
  }
}

TRSM_task::TRSM_task(const Dense& A, Dense& B, int uplo, int lr)
: Task({A}, {B}), args{uplo, lr} {}

void trsm_cpu_func(
  const double* A, uint64_t A_stride,
  double* B, uint64_t B_dim0, uint64_t B_dim1, uint64_t B_stride,
  trsm_args& args
) {
  cblas_dtrsm(
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
  const double* A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  double* B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
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

void TRSM_task::execute() {
  const Dense& A = constant[0];
  Dense& B = modified[0];
  if (schedule_started) {
    struct starpu_task* task = starpu_task_create();
    task->cl = &trsm_cl;
    task->cl_arg = &args;
    task->cl_arg_size = sizeof(args);
    task->handles[0] = starpu_data_lookup(&A);
    task->handles[1] = starpu_data_lookup(&B);
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "trsm_task");
  } else {
    trsm_cpu_func(&A, A.stride, &B, B.dim[0], B.dim[1], B.stride, args);
  }
}

GEMM_task::GEMM_task(
  const Dense& A, const Dense& B, Dense& C,
  bool TransA, bool TransB, double alpha, double beta
) : Task({A, B}, {C}), args{TransA, TransB, alpha, beta} {}

void gemm_cpu_func(
  const double* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  const double* B, uint64_t, uint64_t B_dim1, uint64_t B_stride,
  double* C, uint64_t C_dim0, uint64_t C_dim1, uint64_t C_stride,
  gemm_args& args
) {
  if (B_dim1 == 1) {
    cblas_dgemv(
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
    cblas_dgemm(
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
  const double* A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t A_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  const double* B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
  uint64_t B_dim0 = STARPU_MATRIX_GET_NY(buffers[1]);
  uint64_t B_dim1 = STARPU_MATRIX_GET_NX(buffers[1]);
  uint64_t B_stride = STARPU_MATRIX_GET_LD(buffers[1]);
  double* C = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
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

void GEMM_task::execute() {
  const Dense& A = constant[0];
  const Dense& B = constant[1];
  Dense& C = modified[0];
  if (schedule_started) {
    struct starpu_task* task = starpu_task_create();
    task->cl = &gemm_cl;
    task->cl_arg = &args;
    task->cl_arg_size = sizeof(args);
    task->handles[0] = starpu_data_lookup(&A);
    task->handles[1] = starpu_data_lookup(&B);
    task->handles[2] = starpu_data_lookup(&C);
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

SVD_task::SVD_task(Dense& A, Dense& U, Dense& S, Dense& V)
: Task({}, {A, U, S, V}) {}

void svd_cpu_func(
  double* A, uint64_t A_dim0, uint64_t A_dim1, uint64_t A_stride,
  double* U, uint64_t, uint64_t, uint64_t U_stride,
  double* S, uint64_t S_dim0, uint64_t, uint64_t S_stride,
  double* V, uint64_t, uint64_t, uint64_t V_stride
) {
  std::vector<double> Sdiag(S_dim0, 0);
  std::vector<double> work(S_dim0-1, 0);
  LAPACKE_dgesvd(
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
  double* A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
  uint64_t A_dim0 = STARPU_MATRIX_GET_NY(buffers[0]);
  uint64_t A_dim1 = STARPU_MATRIX_GET_NX(buffers[0]);
  uint64_t A_stride = STARPU_MATRIX_GET_LD(buffers[0]);
  double* U = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
  uint64_t U_dim0 = STARPU_MATRIX_GET_NY(buffers[1]);
  uint64_t U_dim1 = STARPU_MATRIX_GET_NX(buffers[1]);
  uint64_t U_stride = STARPU_MATRIX_GET_LD(buffers[1]);
  double* S = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
  uint64_t S_dim0 = STARPU_MATRIX_GET_NY(buffers[2]);
  uint64_t S_dim1 = STARPU_MATRIX_GET_NX(buffers[2]);
  uint64_t S_stride = STARPU_MATRIX_GET_LD(buffers[2]);
  double* V = (double *)STARPU_MATRIX_GET_PTR(buffers[3]);
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

void SVD_task::execute() {
  Dense& A = modified[0];
  Dense& U = modified[1];
  Dense& S = modified[2];
  Dense& V = modified[3];
  if (schedule_started) {
    struct starpu_task* task = starpu_task_create();
    task->cl = &svd_cl;
    task->handles[0] = starpu_data_lookup(&A);
    task->handles[1] = starpu_data_lookup(&U);
    task->handles[2] = starpu_data_lookup(&S);
    task->handles[3] = starpu_data_lookup(&V);
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

Recompress_col_task::Recompress_col_task(
  Dense& newU, const Dense& AU, const Dense& BU, Dense& AS, const Dense& BS
) : Task({AU, BU, BS}, {newU, AS}) {}

void Recompress_col_task::execute() {
  Dense& AU = constant[0];
  Dense& BU = constant[1];
  Hierarchical US(1, modified.size()-1);
  for (uint64_t i=1; i<modified.size(); ++i) {
    // TODO Assumes equal size of Ss
    US[i-1] = gemm(AU, modified[i]);
    gemm(BU, constant[i+1], US[i-1], 1, 1);
  }
  Dense newU, S, V, USD(US);
  std::tie(newU, S, V) = svd(USD);
  Copy_task(newU, modified[0]).execute();
  Dense SV = gemm(
    resize(S, AU.dim[1], AU.dim[1]), resize(V, AU.dim[1], V.dim[1])
  );
  Hierarchical smallVH(SV, 1, modified.size()-1, false);
  for (uint64_t i=1; i<modified.size(); ++i) {
    Copy_task(Dense(std::move(smallVH[i-1])), modified[i]).execute();
  }
}

Recompress_row_task::Recompress_row_task(
  Dense& newV, const Dense& AV, const Dense& BV, Dense& AS, const Dense& BS
) : Task({AV, BV, BS}, {newV, AS}) {}

void Recompress_row_task::execute() {
  Dense& AV = constant[0];
  Dense& BV = constant[1];
  Hierarchical SV(modified.size()-1, 1);
  for (uint64_t i=1; i<modified.size(); ++i) {
    // TODO Assumes equal size of Ss
    SV[i-1] = gemm(modified[i], AV);
    gemm(constant[i+1], BV, SV[i-1], 1, 1);
  }
  Dense U, S, newV, SVD(SV);
  std::tie(U, S, newV) = svd(SVD);
  Copy_task(newV, modified[0]).execute();
  Dense US = gemm(
    resize(U, U.dim[0], AV.dim[0]), resize(S, AV.dim[0], AV.dim[0])
  );
  Hierarchical USH(US, modified.size()-1, 1, false);
  for (uint64_t i=1; i<modified.size(); ++i) {
    Copy_task(Dense(std::move(USH[i-1])), modified[i]).execute();
  }
}

void add_kernel_task(
  void (*kernel)(
    double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
    const std::vector<std::vector<double>>& x,
    int64_t row_start, int64_t col_start
  ),
  Dense& A, const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
) {
  add_task(std::make_shared<Kernel_task>(kernel, A, x, row_start, col_start));
}

void add_copy_task(
  const Dense& A, Dense& B, int64_t row_start, int64_t col_start
) {
  add_task(std::make_shared<Copy_task>(A, B, row_start, col_start));
}

void add_assign_task(Dense& A, double value) {
  add_task(std::make_shared<Assign_task>(A, value));
}

void add_addition_task(Dense& A, const Dense& B) {
  add_task(std::make_shared<Addition_task>(A, B));
}

void add_subtraction_task(Dense& A, const Dense& B) {
  add_task(std::make_shared<Subtraction_task>(A, B));
}

void add_multiplication_task(Dense& A, double factor) {
  add_task(std::make_shared<Multiplication_task>(A, factor));
}

void add_getrf_task(Dense& AU, Dense& L) {
  add_task(std::make_shared<GETRF_task>(AU, L));
}

void add_qr_task(Dense& A, Dense& Q, Dense& R) {
  // TODO Check for duplicate/shared tasks
  add_task(std::make_shared<QR_task>(A, Q, R));
}

void add_rq_task(Dense& A, Dense& R, Dense& Q) {
  // TODO Check for duplicate/shared tasks
  add_task(std::make_shared<RQ_task>(A, R, Q));
}

void add_trsm_task(const Dense& A, Dense& B, int uplo, int lr) {
  if (!matrix_is_tracked("trsm_task", A, B)) {
    add_task(std::make_shared<TRSM_task>(A, B, uplo, lr));
    register_matrix("trsm_task", A, B);
  }
}

BasisTracker<
  BasisKey, BasisTracker<BasisKey, std::shared_ptr<GEMM_task>>
> gemm_tracker;

void add_gemm_task(
  const Dense& A, const Dense& B, Dense& C,
  bool TransA, bool TransB, double alpha, double beta
) {
  // TODO Only add relevant gemm tasks to tracker?
  // TODO Track tasks objects themselves?
  if (
    beta == 0
    && schedule_started
    && !C.is_submatrix()
    && gemm_tracker.has_key(A) && gemm_tracker[A].has_key(B)
    && gemm_tracker[A][B]->args.alpha == alpha
    && gemm_tracker[A][B]->args.TransA == TransA
    && gemm_tracker[A][B]->args.TransB == TransB
  ) {
    C = gemm_tracker[A][B]->modified[0].share();
    return;
  }
  std::shared_ptr<GEMM_task> task = std::make_shared<GEMM_task>(
    A, B, C, TransA, TransB, alpha, beta
  );
  if (beta == 0 && schedule_started && !C.is_submatrix()) {
    gemm_tracker[A][B] = task;
  }
  add_task(task);
}

void add_svd_task(Dense& A, Dense& U, Dense& S, Dense& V) {
  // TODO Check for duplicate/shared tasks
  add_task(std::make_shared<SVD_task>(A, U, S, V));
}

BasisTracker<
  BasisKey, BasisTracker<BasisKey, std::shared_ptr<Recompress_col_task>>
> recompress_col_tracker;

void add_recompress_col_task(
  Dense& newU, const Dense& AU, const Dense& BU, Dense& AS, const Dense& BS
) {
  assert(schedule_started);
  if (
    recompress_col_tracker.has_key(AU) && recompress_col_tracker[AU].has_key(BU)
  ) {
    recompress_col_tracker[AU][BU]->modified.push_back(AS.share());
    recompress_col_tracker[AU][BU]->constant.push_back(BS.share());
    newU = recompress_col_tracker[AU][BU]->modified[0].share();
  } else {
    recompress_col_tracker[AU][BU] = std::make_shared<Recompress_col_task>(
      newU, AU, BU, AS, BS
    );
    add_task(recompress_col_tracker[AU][BU]);
  }
}

BasisTracker<
  BasisKey, BasisTracker<BasisKey, std::shared_ptr<Recompress_row_task>>
> recompress_row_tracker;

void add_recompress_row_task(
  Dense& newV, const Dense& AV, const Dense& BV, Dense& AS, const Dense& BS
) {
  assert(schedule_started);
  if (
    recompress_row_tracker.has_key(AV) && recompress_row_tracker[AV].has_key(BV)
  ) {
    recompress_row_tracker[AV][BV]->modified.push_back(AS.share());
    recompress_row_tracker[AV][BV]->constant.push_back(BS.share());
    newV = recompress_row_tracker[AV][BV]->modified[0].share();
  } else {
    recompress_row_tracker[AV][BV] = std::make_shared<Recompress_row_task>(
      newV, AV, BV, AS, BS
    );
    add_task(recompress_row_tracker[AV][BV]);
  }
}

void start_schedule() {
  assert(!schedule_started);
  assert(tasks.empty());
  schedule_started = true;
}

void execute_schedule() {
  for (decltype(tasks)::iterator it=tasks.begin(); it!=tasks.end(); ++it) {
    (**it).execute();
  }
  starpu_task_wait_for_all();
  tasks.clear();
  schedule_started = false;
}

void initialize_starpu() {
  STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "init");
  make_kernel_codelet();
  make_copy_codelet();
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
  recompress_col_tracker.clear();
  recompress_row_tracker.clear();
}

} // namespace hicma
