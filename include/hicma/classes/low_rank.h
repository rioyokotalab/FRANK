#ifndef hicma_classes_low_rank_h
#define hicma_classes_low_rank_h

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"

#include <array>
#include <cstdint>
#include <memory>


namespace hicma
{

class LowRank : public Matrix {
 private:
  std::shared_ptr<Dense> _U = std::make_shared<Dense>();
  std::shared_ptr<Dense> _V = std::make_shared<Dense>();
  friend class BasisCopyTracker;
  Dense _S;
 public:
  std::array<int64_t, 2> dim = {0, 0};
  int64_t rank = 0;

  // Special member functions
  LowRank() = default;

  virtual ~LowRank() = default;

  LowRank(const LowRank& A);

  LowRank& operator=(const LowRank& A);

  LowRank(LowRank&& A) = default;

  LowRank& operator=(LowRank&& A) = default;

  // Getters and setters
  Dense& U();
  const Dense& U() const;

  Dense& S();
  const Dense& S() const;

  Dense& V();
  const Dense& V() const;

  // Additional constructors
  LowRank(int64_t n_rows, int64_t n_cols, int64_t k);

  LowRank(const Dense& A, int64_t k);

  LowRank(const Dense& U, const Dense& S, const Dense& V);

  LowRank(std::shared_ptr<Dense> U, const Dense& S, std::shared_ptr<Dense> V);

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
