#ifndef hicma_classes_dense_h
#define hicma_classes_dense_h

#include "hicma/classes/node.h"

#include <array>
#include <cstdint>
#include <vector>


namespace hicma
{

class Dense : public Matrix {
 public:
  std::array<int64_t, 2> dim = {0, 0};
  int64_t stride = 0;
 private:
  std::vector<double> data;
  double* data_ptr = nullptr;
  const double* const_data_ptr = nullptr;
  bool owning = true;

 protected:
  virtual double* get_pointer();

  virtual const double* get_pointer() const;

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
    int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
    Dense& A
  );

  Dense(
    int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
    const Dense& A
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

  // Get part of other Dense
  Dense get_part(
    int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start
  ) const;
};

} // namespace hicma

#endif // hicma_classes_dense_h
