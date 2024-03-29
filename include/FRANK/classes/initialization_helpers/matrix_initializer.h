/**
 * @file matrix_initializer.h
 * @brief Include the `MatrixInitializer` class and a small standard library
 * extension
 *
 * The extension to the C++ standard library allows using FRANK classes with
 * some standard library containers.
 *
 * @copyright Copyright (c) 2020
 */
#ifndef FRANK_classes_initialization_helpers_matrix_initializer_h
#define FRANK_classes_initialization_helpers_matrix_initializer_h

#include "FRANK/definitions.h"
#include "FRANK/classes/dense.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/initialization_helpers/index_range.h"

#include <cstdint>
#include <tuple>
#include <vector>


/**
 * @brief General namespace of the FRANK library
 */
namespace FRANK
{

class ClusterTree;

/**
 * @brief Abstract base class used for initializing submatrices during
 * `Hierarchical` construction
 *
 * While this class implements methods for constructing a (nested) basis as well
 * as for the compression of admissible blocks, its member function
 * `fill_dense_representation()`, used to assign matrix elements, is a pure
 * virtual function. Subclasses of this class thus need to implement this
 * function with a way of retrieving matrix elements. Common examples for this
 * are retrieving them from a kernel function and parameters, reading them from
 * a `Dense` matrix instance or loading them from a file.
 *
 * All other methods of this class internally rely on
 * `fill_dense_representation()`, so implementing this single function is
 * sufficient to fully implement the initializer.
 */
class MatrixInitializer {
 protected:
  const double admis;
  const double eps;
  const int64_t rank;
  const std::vector<std::vector<double>> params;
  const AdmisType admis_type;

  void find_admissible_blocks(const ClusterTree& node);

 public:
  // Special member functions
  MatrixInitializer() = delete;

  ~MatrixInitializer() = default;

  MatrixInitializer(const MatrixInitializer& A) = delete;

  MatrixInitializer& operator=(const MatrixInitializer& A) = delete;

  MatrixInitializer(MatrixInitializer&& A) = delete;

  MatrixInitializer& operator=(MatrixInitializer&& A) = default;

  /**
   * @brief Construct a new `MatrixInitializer`
   *
   * @param admis
   * Distance-to-diagonal or standard admissibility condition constant.
   * @param eps
   * Fixed error threshold used for approximating admissible submatrices.
   * @param rank
   * Fixed rank to be used for approximating admissible submatrices. Ignored if eps &ne; 0
   * @param params
   * Vector containing parameters used as input to the kernel
   * and as coordinate of particles
   * @param admis_type
   * Either AdmisType::PositionBased (Default) or AdmisType::GeometryBased
   */
  MatrixInitializer(
    const double admis, const double eps, const int64_t rank,
    const std::vector<std::vector<double>> params={},
    const AdmisType admis_type=AdmisType::PositionBased
  );

  /**
   * @brief Pure virtual function for assigning matrix elements
   *
   * @param A
   * Matrix whose elements are to be assigned.
   * @param row_range
   * Row range of \p A. The start of the `IndexRange` within the root
   * level `Hierarchical` matrix.
   * @param col_range
   * Column range of \p A. The start of the `IndexRange` within the root
   * level `Hierarchical` matrix.
   *
   * Aside from a constructor, deriving classes need to implement this single
   * method so that it assigns the desired elements to the matrix \p A. How
   * assignment works is up to the user, but it is recommended to implement it
   * as a StarPU task so that task parallelism can be used.
   */
  virtual void fill_dense_representation(
    Dense& A, const IndexRange& row_range, const IndexRange& col_range
  ) const = 0;

  /**
   * @brief Convenience method to get a `Dense` matrix representing a node in a
   * `ClusterTree`
   *
   * @param node
   * `ClusterTree` node to be represented by a `Dense` matrix.
   * @return Dense
   * The `Dense` matrix representing \p node.
   *
   * Internally, this method uses `fill_dense_representation()` to assign the
   * elements of the newly created `Dense` matrix.
   */
  Dense get_dense_representation(const ClusterTree& node) const;

  /**
   * @brief Get a compressed representation of an admissible `ClusterTree` node
   *
   * @param node
   * `ClusterTree` node to be represented by a `LowRank` approximation.
   * @param fixed_rank
   * Whether to use fixed rank for the compression (`true`) or use fixed accuracy/threshold (`false`).
   * @return LowRank
   * `LowRank` approximation representing \p node.
   */
  LowRank get_compressed_representation(const ClusterTree& node, const bool fixed_rank) const;

  /**
   * @brief Check if a `ClusterTree` node is admissible
   *
   * @param node
   * `ClusterTree` node for which admissibility is to be checked.
   * @return true
   * If the node is admissible and can be compressed.
   * @return false
   * If the node is not admissible.
   */
  bool is_admissible(const ClusterTree& node) const;

  virtual std::vector<std::vector<double>> get_coords_range(const IndexRange& range) const;

};

} // namespace FRANK


#endif // FRANK_classes_initialization_helpers_matrix_initializer_h
