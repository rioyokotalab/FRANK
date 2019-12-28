#ifndef hicma_classes_uniform_hierarchical_h
#define hicma_classes_uniform_hierarchical_h

#include "hicma/classes/hierarchical.h"

#include <memory>
#include <vector>

#include "yorel/multi_methods.hpp"

namespace hicma
{

class NodeProxy;
class Dense;

class UniformHierarchical : public Hierarchical {
private:
  std::vector<std::shared_ptr<Dense>> row_basis, col_basis;
public:
  MM_CLASS(UniformHierarchical, Hierarchical);

  // Special member functions
  UniformHierarchical();

  ~UniformHierarchical();

  UniformHierarchical(const UniformHierarchical& A);

  UniformHierarchical& operator=(const UniformHierarchical& A);

  UniformHierarchical(UniformHierarchical&& A);

  UniformHierarchical& operator=(UniformHierarchical&& A);

  // Overridden functions from Hierarchical
  std::unique_ptr<Node> clone() const override;

  std::unique_ptr<Node> move_clone() override;

  const char* type() const override;

  // Additional constructors
  UniformHierarchical(
    void (*func)(
      std::vector<double>& data,
      std::vector<double>& x,
      int ni, int nj,
      int i_begin, int j_begin
    ),
    std::vector<double>& x,
    int ni, int nj,
    int rank,
    int nleaf,
    int admis=1,
    int ni_level=2, int nj_level=2,
    int i_begin=0, int j_begin=0,
    int i_abs=0, int j_abs=0,
    int level=0
  );
};

} // namespace hicma

#endif // hicma_classes_uniform_hierarchical_h
