#ifndef hicma_classes_initialization_helpers_basis_copy_tracker_h
#define hicma_classes_initialization_helpers_basis_copy_tracker_h

#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix_proxy.h"

#include <array>
#include <cstdint>
#include <unordered_map>
#include <memory>


namespace hicma
{

class Dense;
class Matrix;

class BasisCopyTracker {
 private:
  std::unordered_map<std::shared_ptr<Matrix>, MatrixProxy> copied_col_bases;
  std::unordered_map<std::shared_ptr<Matrix>, MatrixProxy> copied_row_bases;
 public:
  // Special member functions
  BasisCopyTracker() = default;

  ~BasisCopyTracker() = default;

  BasisCopyTracker(const BasisCopyTracker& A) = delete;

  BasisCopyTracker& operator=(const BasisCopyTracker& A) = delete;

  BasisCopyTracker(BasisCopyTracker&& A) = delete;

  BasisCopyTracker& operator=(BasisCopyTracker&& A) = delete;

  // Utility methods
  LowRank tracked_copy(const LowRank& A);
};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_basis_copy_tracker_h
