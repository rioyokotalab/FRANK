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
) : Task({A}, {B}), row_start(row_start), col_start(col_start) {}

void Copy_task::execute() {
  const Dense& A = constant[0];
  Dense& B = modified[0];
  if (row_start == 0 && col_start == 0) {
    for (int64_t i=0; i<B.dim[0]; i++) {
      for (int64_t j=0; j<B.dim[1]; j++) {
        B(i, j) = A(i, j);
      }
    }
  } else {
    for (int64_t i=0; i<B.dim[0]; i++) {
      for (int64_t j=0; j<B.dim[1]; j++) {
        B(i, j) = A(row_start+i, col_start+j);
      }
    }
  }
}

Assign_task::Assign_task(Dense& A, double value)
: Task({}, {A}), value(value) {}

void Assign_task::execute() {
  Dense& A = modified[0];
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) = value;
    }
  }
}

Addition_task::Addition_task(Dense& A, const Dense& B)
: Task({B}, {A}) {}

void Addition_task::execute() {
  Dense& A = modified[0];
  const Dense& B = constant[0];
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) += B(i, j);
    }
  }
}

Subtraction_task::Subtraction_task(Dense& A, const Dense& B)
: Task({B}, {A}) {}

void Subtraction_task::execute() {
  Dense& A = modified[0];
  const Dense& B = constant[0];
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) -= B(i, j);
    }
  }
}

Multiplication_task::Multiplication_task(Dense& A, double factor)
: Task({}, {A}), factor(factor) {}

void Multiplication_task::execute() {
  Dense& A = modified[0];
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) *= factor;
    }
  }
}

GETRF_task::GETRF_task(Dense& AU, Dense& L) : Task({}, {AU, L}) {}

void GETRF_task::execute() {
  Dense& AU = modified[0];
  Dense& L = modified[1];
  std::vector<int> ipiv(std::min(AU.dim[0], AU.dim[1]));
  LAPACKE_dgetrf(
    LAPACK_ROW_MAJOR,
    AU.dim[0], AU.dim[1],
    &AU, AU.stride,
    &ipiv[0]
  );
  for (int64_t i=0; i<AU.dim[0]; i++) {
    for (int64_t j=0; j<i; j++) {
      L(i, j) = AU(i, j);
      AU(i, j) = 0;
    }
    L(i, i) = 1;
  }
}

QR_task::QR_task(Dense& A, Dense& Q, Dense& R) : Task({}, {A, Q, R}) {}

void QR_task::execute() {
  Dense& A = modified[0];
  Dense& Q = modified[1];
  Dense& R = modified[2];
  int64_t k = std::min(A.dim[0], A.dim[1]);
  std::vector<double> tau(k);
  for (int64_t i=0; i<std::min(Q.dim[0], Q.dim[1]); i++) Q(i, i) = 1.0;
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A, A.stride, &tau[0]);
  // TODO Consider using A for the dorgqr and moving to Q afterwards! That
  // also simplify this loop.
  for(int64_t i=0; i<Q.dim[0]; i++) {
    for(int64_t j=i; j<Q.dim[1]; j++) {
      R(i, j) = A(i, j);
    }
  }
  for(int64_t i=0; i<Q.dim[0]; i++) {
    for(int64_t j=0; j<std::min(i, Q.dim[1]); j++) {
      Q(i, j) = A(i, j);
    }
  }
  // TODO Consider making special function for this. Performance heavy
  // and not always needed. If Q should be applied to something, use directly!
  // Alternatively, create Dense deriative that remains in elementary
  // reflector form, uses dormqr instead of gemm and can be transformed to
  // Dense via dorgqr!
  LAPACKE_dorgqr(
    LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], k, &Q, Q.stride, &tau[0]
  );
}

RQ_task::RQ_task(Dense& A, Dense& R, Dense& Q) : Task({}, {A, R, Q}) {}

void RQ_task::execute() {
  Dense& A = modified[0];
  Dense& R = modified[1];
  Dense& Q = modified[2];
  int64_t k = std::min(A.dim[0], A.dim[1]);
  std::vector<double> tau(k);
  LAPACKE_dgerqf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A, A.stride, &tau[0]);
  // TODO Consider making special function for this. Performance heavy and not
  // always needed. If Q should be applied to something, use directly!
  // Alternatively, create Dense deriative that remains in elementary reflector
  // form, uses dormqr instead of gemm and can be transformed to Dense via
  // dorgqr!
  for (int64_t i=0; i<R.dim[0]; i++) {
    for (int64_t j=i; j<R.dim[1]; j++) {
      R(i, j) = A(i, A.dim[1]-R.dim[1]+j);
    }
  }
  for(int64_t i=0; i<Q.dim[0]; i++) {
    for(int64_t j=0; j<std::min(A.dim[1]-R.dim[1]+i, Q.dim[1]); j++) {
      Q(i, j) = A(i, j);
    }
  }
  LAPACKE_dorgrq(
    LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], k, &Q, Q.stride, &tau[0]
  );
}

TRSM_task::TRSM_task(const Dense& A, Dense& B, int uplo, int lr)
: Task({A}, {B}), uplo(uplo), lr(lr) {}

void TRSM_task::execute() {
  const Dense& A = constant[0];
  Dense& B = modified[0];
  cblas_dtrsm(
    CblasRowMajor,
    lr==TRSM_LEFT?CblasLeft:CblasRight,
    uplo==TRSM_UPPER?CblasUpper:CblasLower,
    CblasNoTrans,
    uplo==TRSM_UPPER?CblasNonUnit:CblasUnit,
    B.dim[0], B.dim[1],
    1,
    &A, A.stride,
    &B, B.stride
  );
}

GEMM_task::GEMM_task(
  const Dense& A, const Dense& B, Dense& C,
  bool TransA, bool TransB, double alpha, double beta
) : Task({A, B}, {C}),
    TransA(TransA), TransB(TransB), alpha(alpha), beta(beta) {}

void GEMM_task::execute() {
  const Dense& A = constant[0];
  const Dense& B = constant[1];
  Dense& C = modified[0];
  if (B.dim[1] == 1) {
    cblas_dgemv(
      CblasRowMajor,
      CblasNoTrans,
      A.dim[0], A.dim[1],
      alpha,
      &A, A.stride,
      &B, B.stride,
      beta,
      &C, B.stride
    );
  }
  else {
    int64_t k = TransA ? A.dim[0] : A.dim[1];
    cblas_dgemm(
      CblasRowMajor,
      TransA?CblasTrans:CblasNoTrans, TransB?CblasTrans:CblasNoTrans,
      C.dim[0], C.dim[1], k,
      alpha,
      &A, A.stride,
      &B, B.stride,
      beta,
      &C, C.stride
    );
  }
}

SVD_task::SVD_task(Dense& A, Dense& U, Dense& S, Dense& V)
: Task({}, {A, U, S, V}) {}

void SVD_task::execute() {
  Dense& A = modified[0];
  Dense& U = modified[1];
  Dense& S = modified[2];
  Dense& V = modified[3];
  Dense Sdiag(S.dim[0], 1);
  Dense work(S.dim[0]-1, 1);
  LAPACKE_dgesvd(
    LAPACK_ROW_MAJOR,
    'S', 'S',
    A.dim[0], A.dim[1],
    &A, A.stride,
    &Sdiag,
    &U, U.stride,
    &V, V.stride,
    &work
  );
  for(int64_t i=0; i<S.dim[0]; i++){
    S(i, i) = Sdiag[i];
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
  // TODO Check for duplicate/shared tasks
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
    && gemm_tracker[A][B]->alpha == alpha
    && gemm_tracker[A][B]->TransA == TransA
    && gemm_tracker[A][B]->TransB == TransB
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
  schedule_started = false;
  while (!tasks.empty()) {
    tasks.front()->execute();
    tasks.pop_front();
  }
}

} // namespace hicma
