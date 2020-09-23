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
 public:
  MatrixProxy sub_bases;
  Dense translation;

  // TODO This is a stop-gap solution. Maybe we should generally store Vt
  // instead of V to make this unnecessary!
  bool col_basis;

  // Special member functions
  NestedBasis() = default;

  virtual ~NestedBasis() = default;

  NestedBasis(const NestedBasis& A) = default;

  NestedBasis& operator=(const NestedBasis& A) = default;

  NestedBasis(NestedBasis&& A) = default;

  NestedBasis& operator=(NestedBasis&& A) = default;

  // Constructors
  NestedBasis(
    const MatrixProxy& sub_basis, const Dense& translation,
    bool col_basis
  );

  // Utility methods
  bool is_shared_with(const NestedBasis& A) const;

  bool is_col_basis() const;

  bool is_row_basis() const;
};

MatrixProxy share_basis(const Matrix& A);

bool is_shared(const Matrix& A, const Matrix& B);

} // namespace hicma

#endif // hicma_classes_nested_basis_h
