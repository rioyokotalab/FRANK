#ifndef hicma_classes_nested_basis_h
#define hicma_classes_nested_basis_h

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>


namespace hicma
{

class NestedBasis : public Matrix {
 private:
  std::vector<MatrixProxy> sub_bases;
 public:
  Dense transfer_matrix;

  // TODO This is a stop-gap solution. Maybe we should generally store Vt
  // instead of V to make this unnecessary!
  bool col_basis;

  // Special member functions
  NestedBasis() = default;

  virtual ~NestedBasis() = default;

  NestedBasis(const NestedBasis& A);

  NestedBasis& operator=(const NestedBasis& A);

  NestedBasis(NestedBasis&& A) = default;

  NestedBasis& operator=(NestedBasis&& A) = default;

  // Constructors
  NestedBasis(
    Dense&& representation,
    std::vector<MatrixProxy>& sub_bases,
    bool is_col_basis
  );

  NestedBasis(const Dense& representation, bool is_col_basis);

  // Utility methods
  MatrixProxy& operator[](int64_t i);

  const MatrixProxy& operator[](int64_t i) const;

  int64_t num_child_basis() const;

  NestedBasis share() const;

  bool is_shared_with(const NestedBasis& A) const;

  bool is_col_basis() const;

  bool is_row_basis() const;
};

MatrixProxy share_basis(const Matrix& A);

bool is_shared(const Matrix& A, const Matrix& B);

} // namespace hicma

#endif // hicma_classes_nested_basis_h
