#ifndef hicma_classes_initialization_helpers_basis_tracker_h
#define hicma_classes_initialization_helpers_basis_tracker_h

#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/classes/shared_basis.h"
#include "hicma/classes/intitialization_helpers/index_range.h"

#include <cstdint>
#include <functional>
#include <tuple>

namespace std {
  template <>
  struct hash<hicma::IndexRange> {
    size_t operator()(const hicma::IndexRange& key) const;
  };
}

namespace hicma
{

bool operator==(const IndexRange& A, const IndexRange& B);

template<class T>
class BasisTracker {
 private:
  std::unordered_map<T, MatrixProxy> bases;
 public:
  bool has_basis(const T& key) const {
    return (bases.find(key) != bases.end());
  }

  void register_basis(const T& key, const Matrix& A=Matrix()) {
    bases[key] = A;
  }

  void register_basis(const T& key, MatrixProxy&& A) {
    bases[key] = std::move(A);
  }

  const MatrixProxy& operator[](const T& key) const {
    return bases[key];
  }

  MatrixProxy& operator[](const T& key) {
    return bases[key];
  }

  MatrixProxy get_shared(const T& key) const {
    return share_basis(bases[key]);
  }
};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_basis_tracker_h
