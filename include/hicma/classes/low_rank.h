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
 public:
  std::array<int64_t, 2> dim = {0, 0};
  int64_t rank = 0;
  Dense U, S, V;

  // Special member functions
  LowRank() = default;

  virtual ~LowRank() = default;

  LowRank(const LowRank& A) = default;

  LowRank& operator=(const LowRank& A) = default;

  LowRank(LowRank&& A) = default;

  LowRank& operator=(LowRank&& A) = default;

  // Implicit conversion from temporaries, requires them to actually be LR
  LowRank(MatrixProxy&& A);

  // Additional constructors
  LowRank(int64_t n_rows, int64_t n_cols, int64_t k);

  LowRank(const Dense& A, int64_t k);

  LowRank(const Matrix& U, const Dense& S, const Matrix& V, bool copy=true);

  LowRank(Dense&& U, Dense&& S, Dense&& V);
};

} // namespace hicma

#endif // hicma_classes_low_rank_h
