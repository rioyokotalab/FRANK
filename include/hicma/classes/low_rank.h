#ifndef hicma_classes_low_rank_h
#define hicma_classes_low_rank_h

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"

#include <array>
#include <cstdint>


namespace hicma
{

class LowRank : public Matrix {
 private:
  MatrixProxy _U, _V;
  friend class BasisCopyTracker;
  Dense _S;
 public:
  std::array<int64_t, 2> dim = {0, 0};
  int64_t rank = 0;

  // Special member functions
  LowRank() = default;

  virtual ~LowRank() = default;

  LowRank(const LowRank& A) = default;

  LowRank& operator=(const LowRank& A) = default;

  LowRank(LowRank&& A) = default;

  LowRank& operator=(LowRank&& A) = default;

  // Getters and setters
  MatrixProxy& U();
  const MatrixProxy& U() const;

  Dense& S();
  const Dense& S() const;

  MatrixProxy& V();
  const MatrixProxy& V() const;

  // Additional constructors
  LowRank(int64_t n_rows, int64_t n_cols, int64_t k);

  LowRank(const Dense& A, int64_t k);

  LowRank(const Matrix& U, const Dense& S, const Matrix& V, bool copy_S=false);

  LowRank(
    const LowRank& A,
    int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
    bool copy=false
  );

  // Utility methods
  void mergeU(const LowRank& A, const LowRank& B);

  void mergeS(const LowRank& A, const LowRank& B);

  void mergeV(const LowRank& A, const LowRank& B);
};

} // namespace hicma

#endif // hicma_classes_low_rank_h
