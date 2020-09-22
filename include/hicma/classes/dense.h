#ifndef hicma_classes_dense_h
#define hicma_classes_dense_h

#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>


namespace hicma
{

class DataHandler;
class IndexRange;
class Task;

class Dense : public Matrix {
  // TODO Find way to avoid using friend here! Best not to rely on it.
  // Also don't wanna expose the DataHandler directly though...
  friend Task;
 public:
  std::array<int64_t, 2> dim = {0, 0};
  int64_t stride = 0;
 private:
  std::shared_ptr<DataHandler> data;
  std::array<int64_t, 2> rel_start = {0, 0};

  DataHandler& get_handler();
 protected:
  double* data_ptr = nullptr;
 public:
  // Special member functions
  Dense() = default;

  virtual ~Dense() = default;

  Dense(const Dense& A);

  Dense& operator=(const Dense& A);

  Dense(Dense&& A) = default;

  Dense& operator=(Dense&& A) = default;

  // Explicit conversions using multiple-dispatch function.
  explicit Dense(const Matrix& A);

  // Implicit conversion from temporaries, requires them to actually be D
  Dense(MatrixProxy&& A);

  // Additional constructors
  Dense(int64_t n_rows, int64_t n_cols=1);

  // TODO Add overload where vector doesn't need to be passed. That function
  // should forward to this one with a 0-sized vector. This is to make
  // initialization with functions like identity and random_uniform easier.
  Dense(
    void (*func)(
      double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
      const std::vector<std::vector<double>>& x,
      int64_t row_start, int64_t col_start
    ),
    const std::vector<std::vector<double>>& x,
    int64_t n_rows, int64_t n_cols=1,
    int64_t row_start=0, int64_t col_start=0
  );

  // Additional operators
  const Dense& operator=(const double a);

  double& operator[](int64_t i);

  const double& operator[](int64_t i) const;

  double& operator()(int64_t i, int64_t j);

  const double& operator()(int64_t i, int64_t j) const;

  double* operator&();

  const double* operator&() const;

  // Utility methods
  Dense share() const;

  bool is_shared() const;

  bool is_shared_with(const Dense& A) const;

  bool is_submatrix() const;

  std::vector<Dense> split(
    const std::vector<IndexRange>& row_ranges,
    const std::vector<IndexRange>& col_ranges,
    bool copy=false
  ) const;

  std::vector<Dense> split(
    uint64_t n_row_splits, uint64_t n_col_splits, bool copy=false
  ) const;
};

} // namespace hicma

#endif // hicma_classes_dense_h
