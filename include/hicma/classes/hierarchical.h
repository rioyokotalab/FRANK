#ifndef hicma_classes_hierarchical_h
#define hicma_classes_hierarchical_h

#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>


namespace hicma
{

class BasisKey;
class Dense;
class ClusterTree;
class MatrixInitializer;

class Hierarchical : public Matrix {
 public:
  std::array<int64_t, 2> dim = {0, 0};
 private:
  std::vector<MatrixProxy> data;
 public:
  // Special member functions
  Hierarchical() = default;

  virtual ~Hierarchical() = default;

  Hierarchical(const Hierarchical& A) = default;

  Hierarchical& operator=(const Hierarchical& A) = default;

  Hierarchical(Hierarchical&& A) = default;

  Hierarchical& operator=(Hierarchical&& A) = default;

  // Implicit conversion from temporaries, requires them to actually be H
  Hierarchical(MatrixProxy&&);

  // Additional constructors
  Hierarchical(int64_t n_row_blocks, int64_t n_col_blocks=1);

  Hierarchical(
    const ClusterTree& node,
    MatrixInitializer& initer
  );

  Hierarchical(
    void (*func)(
      float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
      const std::vector<std::vector<float>>& x,
      int64_t row_start, int64_t col_start
    ),
    const std::vector<std::vector<float>>& x,
    int64_t n_rows, int64_t n_cols,
    int64_t rank,
    int64_t nleaf,
    int64_t admis=1,
    int64_t n_row_blocks=2, int64_t n_col_blocks=2,
    int64_t row_start=0, int64_t col_start=0
  );

  Hierarchical(
    Dense&& A,
    int64_t rank,
    int64_t nleaf,
    int64_t admis=1,
    int64_t n_row_blocks=2, int64_t n_col_blocks=2,
    int64_t row_start=0, int64_t col_start=0
  );

  // Additional operators
  const MatrixProxy& operator[](int64_t i) const;

  MatrixProxy& operator[](int64_t i);

  const MatrixProxy& operator[](const ClusterTree& node) const;

  MatrixProxy& operator[](const ClusterTree& node);

  const MatrixProxy& operator()(int64_t i, int64_t j) const;

  MatrixProxy& operator()(int64_t i, int64_t j);

};

} // namespace hicma

#endif // hicma_classes_hierarchical_h
