/**
 * @file low_rank.h
 * @brief Include the `LowRank` matrix class.
 *
 * @copyright Copyright (c) 2020
 */
#ifndef hicma_classes_low_rank_h
#define hicma_classes_low_rank_h

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"

#include <array>
#include <cstdint>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

/**
 * @brief Class handling a matrix decomposed into three factors
 *
 *
 */
class LowRank : public Matrix {
 public:
  /**
   * @brief Dimension of the matrix {rows, columns}
   *
   * The number of rows of the full `LowRank` is the same as the number of rows
   * of #U, and the number of columns the same as that of #V.
   */
  std::array<int64_t, 2> dim = {0, 0};
  /**
   * @brief Rank of the low-rank representation
   *
   * Currently, the same rank is used for both the row and column basis. The
   * size of #S is {rank, rank}.
   */
  int64_t rank = 0;
  /**
   * @brief First factor of the decomposed matrix
   *
   * If an SVD decomposition was used to construct the low-rank representation,
   * then #U is the column basis of the compressed matrix.
   */
  Dense U;
  /**
   * @brief Second factor of the decomposed matrix
   *
   * If an SVD decomposition was used to construct the low-rank representation,
   * then #S is a diagonal matrix containing the singular values on its
   * diagonal.
   */
  Dense S;
  /**
   * @brief Third factor of the decomposed matrix
   *
   * If an SVD decomposition was used to construct the low-rank representation,
   * then #V is the row basis of the compressed matrix.
   */
  Dense V;

  // Special member functions
  LowRank() = default;

  virtual ~LowRank() = default;

  LowRank(const LowRank& A) = default;

  LowRank& operator=(const LowRank& A) = default;

  LowRank(LowRank&& A) = default;

  LowRank& operator=(LowRank&& A) = default;

  /**
   * @brief Move from a `MatrixProxy` instance containing a `LowRank` matrix
   *
   * @param A
   * `MatrixProxy` that must contain a `LowRank` instance.
   *
   * This move constructor is useful to move from a sub-matrix of a
   * `Hierarchical` instance that is known to be of `LowRank` type. Passing a
   * `MatrixProxy` to this constructor that does not contain a `LowRank`
   * instance will result in a runtime error.
   *
   * This constructor is intended to be a convenience method for testing code
   * and quick development. If you find yourself using this function often, you
   * should likely refactor your code to use an \OMM instead.
   */
  LowRank(MatrixProxy&& A);

  // Additional constructors
  /**
   * @brief Construct a new `LowRank` object by compressing a `Dense` matrix
   *
   * @param A
   * `Dense` matrix to be compressed
   * @param rank
   * Rank to be used in approximating \p A
   *
   * A randomized SVD is used to factorize \p A into three matrices #U, #S and
   * #V.
   */
  LowRank(const Dense& A, int64_t rank);

  /**
   * @brief Construct a new `LowRank` object from the three factors
   *
   * @param U
   * First factor of the decomposition.
   * @param S
   * Second factor of the decomposition.
   * @param V
   * Third factor of the decomposition.
   * @param copy_S
   * If true, \p S will be deep copied, otherwise a shallow copy will be used.
   *
   * Note that shallow copies resulting in shared data will be made of \p U and
   * \p V, whereas this depends on \p copy_S for \p S.
   */
  LowRank(const Matrix& U, const Dense& S, const Matrix& V, bool copy=true);


  /**
   * @brief Construct a new `LowRank` object from the three factors
   *
   * @param U
   * First factor of the decomposition.
   * @param S
   * Second factor of the decomposition.
   * @param V
   * Third factor of the decomposition.
   *
   * This constructor will move the `Dense` objects passed to it into the new
   * `LowRank`.
   */
  LowRank(Dense&& U, Dense&& S, Dense&& V);
};

} // namespace hicma

#endif // hicma_classes_low_rank_h
