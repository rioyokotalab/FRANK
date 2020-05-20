#ifndef hicma_classis_basis_h
#define hicma_classis_basis_h

#include "hicma/classes/dense.h"

#include <cstdint>
#include <memory>


namespace hicma
{

class Basis {
 private:
  // Use std::shared_ptr<Matrix> instead and allow different representations!
  // Either get rid of LowRank.U() and use conversion operator or have
  // LowRank.U() return Matrix&
  std::shared_ptr<Dense> representation;
 public:
  // Special member functions
  Basis() = default;

  virtual ~Basis() = default;

  Basis(const Basis& A);

  Basis& operator=(const Basis& A);

  Basis(Basis&& A) = default;

  Basis& operator=(Basis&& A) = default;

  // Constructors
  Basis(int64_t n_rows, int64_t n_cols);

  Basis(
    int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
    const Dense& A
  );

  // Additional operators
  Basis& operator=(const Dense& A);

  Basis& operator=(Dense&& A);

  operator Dense&();

  operator const Dense&() const;
};

} // namespace hicma

#endif // hicma_classis_basis_h
