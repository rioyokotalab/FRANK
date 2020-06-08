#ifndef hicma_classes_initialization_helpers_basis_tracker_h
#define hicma_classes_initialization_helpers_basis_tracker_h

#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/classes/shared_basis.h"
#include "hicma/classes/intitialization_helpers/index_range.h"

#include <cstdint>
#include <functional>
#include <tuple>


namespace hicma
{

class BasisKey {
 public:
  const double* data_ptr;
  std::array<int64_t, 2> dim;

  BasisKey(const MatrixProxy&);
};

} // namespace hicma


namespace std {
  template <>
  struct hash<hicma::BasisKey> {
    size_t operator()(const hicma::BasisKey& key) const;
  };

  template <>
  struct hash<hicma::IndexRange> {
    size_t operator()(const hicma::IndexRange& key) const;
  };
}

namespace hicma
{

bool operator==(const BasisKey& A, const BasisKey& B);

bool operator==(const IndexRange& A, const IndexRange& B);

template<class T>
class BasisTracker {
 private:
  std::unordered_map<T, MatrixProxy> bases;
 public:
  bool has_basis(const T& key) const {
    return (bases.find(key) != bases.end());
  }

  const MatrixProxy& operator[](const T& key) const {
    return bases[key];
  }

  MatrixProxy& operator[](const T& key) {
    return bases[key];
  }
};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_basis_tracker_h
