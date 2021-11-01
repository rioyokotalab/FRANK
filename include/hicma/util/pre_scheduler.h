#ifndef hicma_util_pre_scheduler_h
#define hicma_util_pre_scheduler_h

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Dense;
class Task;

void add_task(std::shared_ptr<Task> task);

void add_kernel_task(
  void (*kernel)(
    double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
    const std::vector<std::vector<double>>& x,
    int64_t row_start, int64_t col_start
  ),
  Dense& A, const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

void add_copy_task(
  const Dense& A, Dense& B, int64_t row_start=0, int64_t col_start=0
);

void add_transpose_task(const Dense& A, Dense& B);

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
  double alpha, double beta, bool TransA, bool TransB
);

void add_svd_task(Dense& A, Dense& U, Dense& S, Dense& V);

void start_schedule();

void execute_schedule();

void initialize_starpu();

void start_tracking();

void stop_tracking();

} // namespace hicma

#endif // hicma_util_pre_scheduler_h
