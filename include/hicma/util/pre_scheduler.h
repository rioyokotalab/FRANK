#ifndef hicma_util_pre_scheduler_h
#define hicma_util_pre_scheduler_h

#include <cstdint>
#include <functional>
#include <vector>


#include <string>


namespace hicma
{

class Dense;

class Task {
 public:
  std::vector<Dense> constant;
  std::vector<Dense> modified;

  // Special member functions
  Task() = default;

  virtual ~Task() = default;

  Task(const Task& A) = default;

  Task& operator=(const Task& A) = default;

  Task(Task&& A) = default;

  Task& operator=(Task&& A) = default;

  // Execute the task
  virtual void execute() = 0;

 protected:
  Task(
    std::vector<std::reference_wrapper<const Dense>> constant,
    std::vector<std::reference_wrapper<Dense>> modified
  );
};

class Kernel_task : public Task {
 public:
  void (*kernel)(
    Dense& A,
    const std::vector<std::vector<double>>& x,
    int64_t row_start, int64_t col_start
  ) = nullptr;
  const std::vector<std::vector<double>>& x;
  int64_t row_start, col_start;
  Kernel_task(
    void (*kernel)(
      Dense& A, const std::vector<std::vector<double>>& x,
      int64_t row_start, int64_t col_start
    ),
    Dense& A, const std::vector<std::vector<double>>& x,
    int64_t row_start, int64_t col_start
  );

  void execute() override;
};

class Copy_task : public Task {
 public:
  int64_t row_start, col_start;
  Copy_task(const Dense& A, Dense& B, int64_t row_start=0, int64_t col_start=0);

  void execute() override;
};

class Assign_task : public Task {
 public:
  double value;
  Assign_task(Dense& A, double value);

  void execute() override;
};

class Addition_task : public Task {
 public:
  Addition_task(Dense& A, const Dense& B);

  void execute() override;
};

class Subtraction_task : public Task {
 public:
  Subtraction_task(Dense& A, const Dense& B);

  void execute() override;
};

class Multiplication_task : public Task {
 public:
  double factor;
  Multiplication_task(Dense& A, double factor);

  void execute() override;
};

class GETRF_task : public Task {
 public:
  GETRF_task(Dense& AU, Dense& L);

  void execute() override;
};

class QR_task : public Task {
 public:
  QR_task(Dense& A, Dense& Q, Dense& R);

  void execute() override;
};

class RQ_task : public Task {
 public:
  RQ_task(Dense& A, Dense& R, Dense& Q);

  void execute() override;
};

class TRSM_task : public Task {
 public:
  int uplo, lr;
  TRSM_task(const Dense& A, Dense& B, int uplo, int lr);

  void execute() override;
};

class GEMM_task : public Task {
 public:
  bool TransA, TransB;
  double alpha, beta;
  GEMM_task(
    const Dense& A, const Dense& B, Dense& C,
    bool TransA, bool TransB, double alpha, double beta
  );

  void execute() override;
};

class SVD_task : public Task {
 public:
  SVD_task(Dense& A, Dense& U, Dense& S, Dense& V);

  void execute() override;
};

class Recompress_col_task : public Task {
 public:
  bool is_col;
  Recompress_col_task(
    Dense& newU, const Dense& AU, const Dense& BU, Dense& AS, const Dense& BS
  );

  void execute() override;
};

class Recompress_row_task : public Task {
 public:
  bool is_col;
  Recompress_row_task(
    Dense& newV, const Dense& AV, const Dense& BV, Dense& AS, const Dense& BS
  );

  void execute() override;
};

void add_kernel_task(
  void (*kernel)(
    Dense& A, const std::vector<std::vector<double>>& x,
    int64_t row_start, int64_t col_start
  ),
  Dense& A, const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

void add_copy_task(
  const Dense& A, Dense& B, int64_t row_start=0, int64_t col_start=0
);

void add_assign_task(Dense& A, double value);

void add_addition_task(Dense& A, const Dense& B);

void add_subtraction_task(Dense& A, const Dense& B);

void add_multiplication_task(Dense& A, double factor);

void add_getrf_task(Dense& AU, Dense& L);

void add_qr_task(Dense& A, Dense& Q, Dense& R);

void add_rq_task(Dense& A, Dense& R, Dense& Q);

void add_trsm_task(const Dense& A, Dense& B, int uplo, int lr);

void add_gemm_task(
  const Dense& A, const Dense& B, Dense& C,
  bool TransA, bool TransB, double alpha, double beta
);

void add_svd_task(Dense& A, Dense& U, Dense& S, Dense& V);

void add_recompress_col_task(
  Dense& newU, const Dense& AU, const Dense& BU, Dense& AS, const Dense& BS
);

void add_recompress_row_task(
  Dense& newV, const Dense& AV, const Dense& BV, Dense& AS, const Dense& BS
);

void start_schedule();

void execute_schedule();

} // namespace hicma

#endif // hicma_util_pre_scheduler_h
