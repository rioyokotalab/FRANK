#ifndef hicma_classes_low_rank_h
#define hicma_classes_low_rank_h

#include "hicma/classes/dense.h"
#include "hicma/classes/node.h"

#include <array>
#include <cstdint>


namespace hicma
{

class LowRank : public Node {
 private:
  Dense _U, _S, _V;
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

  // Overridden functions from Node
  virtual const char* type() const override;

  // Getters and setters
  virtual Dense& U();
  virtual const Dense& U() const;

  virtual Dense& S();
  virtual const Dense& S() const;

  virtual Dense& V();
  virtual const Dense& V() const;

  // Additional constructors
  LowRank(int64_t n_rows, int64_t n_cols, int64_t k);

  LowRank(const Dense& A, int64_t k);

  LowRank(const Dense& U, const Dense& S, const Dense& V);

  LowRank(
    int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
    const LowRank& A
  );

  // Utility methods
  void mergeU(const LowRank& A, const LowRank& B);

  void mergeS(const LowRank& A, const LowRank& B);

  void mergeV(const LowRank& A, const LowRank& B);

  LowRank get_part(
    int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start
  ) const;
};

} // namespace hicma

#endif // hicma_classes_low_rank_h
