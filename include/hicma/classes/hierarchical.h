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
template<class T>
class BasisTracker;
class Dense;
class ClusterTree;
class MatrixInitializer;

enum { NORMAL_BASIS, SHARED_BASIS };

class Hierarchical : public Matrix {
 public:
  std::array<int64_t, 2> dim = {0, 0};
 private:
  std::vector<MatrixProxy> data;
 public:
  // Special member functions
  Hierarchical() = default;

  virtual ~Hierarchical() = default;

  Hierarchical(const Hierarchical& A);

  Hierarchical& operator=(const Hierarchical& A);

  Hierarchical(Hierarchical&& A) = default;

  Hierarchical& operator=(Hierarchical&& A) = default;

  // Conversion constructors
  Hierarchical(MatrixProxy&&);

  Hierarchical(
    const Matrix& A, int64_t n_row_blocks, int64_t n_col_blocks, bool copy=true
  );

  Hierarchical(const Matrix& A, const Hierarchical& like, bool copy=true);

  // Additional constructors
  Hierarchical(const Hierarchical& A, BasisTracker<BasisKey>& tracker);

  Hierarchical(int64_t n_row_blocks, int64_t n_col_blocks=1);

  Hierarchical(
    const ClusterTree& node,
    MatrixInitializer& initer
  );

  Hierarchical(
    void (*func)(
      Dense& A, const std::vector<std::vector<double>>& x,
      int64_t row_start, int64_t col_start
    ),
    const std::vector<std::vector<double>>& x,
    int64_t n_rows, int64_t n_cols,
    int64_t rank,
    int64_t nleaf,
    int64_t admis=1,
    int64_t n_row_blocks=2, int64_t n_col_blocks=2,
    int basis_type=NORMAL_BASIS,
    int64_t row_start=0, int64_t col_start=0
  );

  // Additional operators
  const MatrixProxy& operator[](int64_t i) const;

  MatrixProxy& operator[](int64_t i);

  const MatrixProxy& operator[](const ClusterTree& node) const;

  MatrixProxy& operator[](const ClusterTree& node);

  const MatrixProxy& operator()(int64_t i, int64_t j) const;

  MatrixProxy& operator()(int64_t i, int64_t j);

  // Utility methods
  void blr_col_qr(Hierarchical& Q, Hierarchical& R);

  void split_col(Hierarchical& QL);

  void restore_col(const Hierarchical& Sp, const Hierarchical& QL);

  void col_qr(int64_t j, Hierarchical& Q, Hierarchical &R);
};

} // namespace hicma

#endif // hicma_classes_hierarchical_h
