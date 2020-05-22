#ifndef hicma_classes_initialization_helpers_basis_tree_h
#define hicma_classes_initialization_helpers_basis_tree_h

#include <array>
#include <cstdint>
#include <map>
#include <memory>


namespace hicma
{

class Dense;
class LowRank;

class BasisCopyTracker {
 private:
  std::map<std::shared_ptr<Dense>, std::shared_ptr<Dense>> copied_col_bases;
  std::map<std::shared_ptr<Dense>, std::shared_ptr<Dense>> copied_row_bases;
 public:
  // Special member functions
  BasisCopyTracker() = default;

  ~BasisCopyTracker() = default;

  BasisCopyTracker(const BasisCopyTracker& A) = delete;

  BasisCopyTracker& operator=(const BasisCopyTracker& A) = delete;

  BasisCopyTracker(BasisCopyTracker&& A) = delete;

  BasisCopyTracker& operator=(BasisCopyTracker&& A) = delete;

  // Utility methods
  std::shared_ptr<Dense> copy_row_basis(const LowRank& A);

  std::shared_ptr<Dense> copy_col_basis(const LowRank& A);
};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_basis_tree_h
