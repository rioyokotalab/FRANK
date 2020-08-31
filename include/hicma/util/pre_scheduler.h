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

class Resize_task : public Task {
 public:
  int64_t n_rows, n_cols;
  Resize_task(const Dense& A, Dense& resized, int64_t n_rows, int64_t n_cols);

  void execute() override;
};

class QR_task : public Task {
 public:
  QR_task(Dense& A, Dense& Q, Dense& R);

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

void add_kernel_task(
  void (*kernel)(
    Dense& A, const std::vector<std::vector<double>>& x,
    int64_t row_start, int64_t col_start
  ),
  Dense& A, const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

void add_resize_task(
  const Dense& A, Dense& resized, int64_t n_rows, int64_t n_cols
);

void add_qr_task(Dense& A, Dense& Q, Dense& R);

void add_gemm_task(
  const Dense& A, const Dense& B, Dense& C,
  bool TransA, bool TransB, double alpha, double beta
);

void add_svd_task(Dense& A, Dense& U, Dense& S, Dense& V);

void start_schedule();

void execute_schedule();

} // namespace hicma

#endif // hicma_util_pre_scheduler_h
