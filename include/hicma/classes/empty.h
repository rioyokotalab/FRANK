#include "hicma/classes/matrix.h"

#include <cstdint>


namespace hicma
{

class Empty : public Matrix {
 public:
  std::array<int64_t, 2> dim = {0, 0};

  // Special member functions
  Empty() = default;

  virtual ~Empty() = default;

  Empty(const Empty& A) = default;

  Empty& operator=(const Empty& A) = default;

  Empty(Empty&& A) = default;

  Empty& operator=(Empty&& A) = default;

  Empty(int64_t n_rows, int64_t n_cols) : dim{n_rows, n_cols} {}
};

}
