#include "hicma/classes/basis.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>


namespace hicma
{

Basis::Basis(const Basis& A) : representation(std::make_shared<Dense>(A)) {}

Basis& Basis::operator=(const Basis& A) {
  representation = std::make_shared<Dense>(A);
  return *this;
}

Basis::Basis(int64_t n_rows, int64_t n_cols)
: representation(std::make_shared<Dense>(n_rows, n_cols)) {}

Basis::Basis(
  int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
  const Dense& A
) : representation(
  std::make_shared<Dense>(n_rows, n_cols, row_start, col_start, A)
) {}

// Additional operators
Basis& Basis::operator=(const Dense& A) {
  representation = std::make_shared<Dense>(A);
  return *this;
}

Basis& Basis::operator=(Dense&& A) {
  representation = std::make_shared<Dense>(std::move(A));
  return *this;
}

Basis::operator Dense&() {
  assert(representation.get() != nullptr);
  return *representation;
}

Basis::operator const Dense&() const {
  assert(representation.get() != nullptr);
  return *representation;
}


} // namespace hicma
