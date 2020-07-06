#ifndef hicma_classes_shared_basis_h
#define hicma_classes_shared_basis_h

#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"

#include <cstdint>
#include <memory>
#include <vector>


namespace hicma
{

class Dense;

class SharedBasis : public Matrix {
 private:
  std::shared_ptr<Dense> transfer_matrix;
  std::vector<SharedBasis> sub_bases;
 public:
  // Special member functions
  SharedBasis() = default;

  virtual ~SharedBasis() = default;

  SharedBasis(const SharedBasis& A);

  SharedBasis& operator=(const SharedBasis& A);

  SharedBasis(SharedBasis&& A) = default;

  SharedBasis& operator=(SharedBasis&& A) = default;

  // Constructors
  SharedBasis(Dense&& representation);

  SharedBasis(std::shared_ptr<Dense> representation);

  // Utility methods
  SharedBasis& operator[](int64_t i);

  const SharedBasis& operator[](int64_t i) const;

  int64_t num_child_basis() const;

  SharedBasis share() const;

  std::shared_ptr<Dense> get_ptr() const;
};

MatrixProxy share_basis(const Matrix& A);

bool is_shared(const Matrix& A, const Matrix& B);

} // namespace hicma

#endif // hicma_classes_shared_basis_h
