#include "hicma/util/pre_scheduler.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
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

Kernel_task::Kernel_task(
  void (*kernel)(
    Dense& A, const std::vector<std::vector<double>>& x,
    int64_t row_start, int64_t col_start
  ),
  Dense& A, const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
) : Task({}, {A}),
    kernel(kernel), x(x), row_start(row_start), col_start(col_start) {}

void Kernel_task::execute() {
  kernel(modified[0], x, row_start, col_start);
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

Resize_task::Resize_task(
  const Dense& A, Dense& resized, int64_t n_rows, int64_t n_cols
) : Task({A}, {resized}), n_rows(n_rows), n_cols(n_cols) {}

void Resize_task::execute() {
  for (int64_t i=0; i<n_rows; i++) {
    for (int64_t j=0; j<n_cols; j++) {
      modified[0](i, j) = constant[0](i, j);
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

std::list<std::shared_ptr<Task>> tasks;
bool schedule_started = false;

void add_task(std::shared_ptr<Task> task) {
  if (schedule_started) {
    tasks.push_back(task);
  } else {
    task->execute();
  }
}

void add_kernel_task(
  void (*kernel)(
    Dense& A, const std::vector<std::vector<double>>& x,
    int64_t row_start, int64_t col_start
  ),
  Dense& A, const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
) {
  std::shared_ptr<Task> task = std::make_shared<Kernel_task>(
    kernel, A, x, row_start, col_start
  );
  add_task(task);
}

void add_copy_task(
  const Dense& A, Dense& B, int64_t row_start, int64_t col_start
) {
  std::shared_ptr<Task> task = std::make_shared<Copy_task>(
    A, B, row_start, col_start
  );
  add_task(task);
}

void add_assign_task(Dense& A, double value) {
  std::shared_ptr<Task> task = std::make_shared<Assign_task>(A, value);
  add_task(task);
}

void add_resize_task(
  const Dense& A, Dense& resized, int64_t n_rows, int64_t n_cols
) {
  std::shared_ptr<Task> task = std::make_shared<Resize_task>(
    A, resized, n_rows, n_cols
  );
  add_task(task);
}

void add_addition_task(Dense& A, const Dense& B) {
  std::shared_ptr<Task> task = std::make_shared<Addition_task>(A, B);
  add_task(task);
}

void add_subtraction_task(Dense& A, const Dense& B) {
  std::shared_ptr<Task> task = std::make_shared<Subtraction_task>(A, B);
  add_task(task);
}

void add_multiplication_task(Dense& A, double factor) {
  std::shared_ptr<Task> task = std::make_shared<Multiplication_task>(A, factor);
  add_task(task);
}

void add_getrf_task(Dense& AU, Dense& L) {
  // TODO Check for duplicate/shared tasks
  std::shared_ptr<Task> task = std::make_shared<GETRF_task>(AU, L);
  add_task(task);
}

void add_qr_task(Dense& A, Dense& Q, Dense& R) {
  // TODO Check for duplicate/shared tasks
  std::shared_ptr<Task> task = std::make_shared<QR_task>(A, Q, R);
  add_task(task);
}

void add_rq_task(Dense& A, Dense& R, Dense& Q) {
  // TODO Check for duplicate/shared tasks
  std::shared_ptr<Task> task = std::make_shared<RQ_task>(A, R, Q);
  add_task(task);
}

void add_trsm_task(const Dense& A, Dense& B, int uplo, int lr) {
  // TODO Check for duplicate/shared tasks
  std::shared_ptr<Task> task = std::make_shared<TRSM_task>(A, B, uplo, lr);
  add_task(task);
}

void add_gemm_task(
  const Dense& A, const Dense& B, Dense& C,
  bool TransA, bool TransB, double alpha, double beta
) {
  // TODO Check for duplicate/shared tasks
  std::shared_ptr<Task> task = std::make_shared<GEMM_task>(
    A, B, C, TransA, TransB, alpha, beta
  );
  add_task(task);
}

void add_svd_task(Dense& A, Dense& U, Dense& S, Dense& V) {
  // TODO Check for duplicate/shared tasks
  std::shared_ptr<Task> task = std::make_shared<SVD_task>(A, U, S, V);
  add_task(task);
}

void start_schedule() {
  assert(!schedule_started);
  assert(tasks.empty());
  schedule_started = true;
}

void execute_schedule() {
  while (!tasks.empty()) {
    tasks.front()->execute();
    tasks.pop_front();
  }
  schedule_started = false;
}

} // namespace hicma
