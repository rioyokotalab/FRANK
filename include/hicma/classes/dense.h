#ifndef hicma_classes_dense_h
#define hicma_classes_dense_h

#include "hicma/classes/matrix.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>


namespace hicma
{

class Dense : public Matrix {
 public:
  std::array<int64_t, 2> dim = {0, 0};
  int64_t stride = 0;
 private:
  std::shared_ptr<std::vector<double>> data
    = std::make_shared<std::vector<double>>();
  std::array<int64_t, 2> rel_start = {0, 0};
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

  // Additional constructors
  Dense(int64_t n_rows, int64_t n_cols=1);

  // TODO Add overload where vector doesn't need to be passed. That function
  // should forward to this one with a 0-sized vector. This is to make
  // initialization with functions like identity and random_uniform easier.
  Dense(
    void (*func)(
      Dense& A, const std::vector<std::vector<double>>& x,
      int64_t row_start, int64_t col_start
    ),
    const std::vector<std::vector<double>>& x,
    int64_t n_rows, int64_t n_cols=1,
    int64_t row_start=0, int64_t col_start=0
  );

  Dense(
    const Dense& A,
    int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
    bool copy=false
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
  int64_t size() const;

  void resize(int64_t dim0, int64_t dim1);

  Dense transpose() const;

  void transpose();

  bool is_shared() const;

  bool is_shared_with(const Dense& A) const;
};

} // namespace hicma

#endif // hicma_classes_dense_h
