#ifndef hicma_oprations_randomized_factorizations_h
#define hicma_oprations_randomized_factorizations_h

#include <cstdint>
#include <tuple>
#include <vector>


namespace hicma
{

class Dense;

/**
 * @brief Compute randomized one-sided interpolatory decomposition (ID) of a `Dense` matrix
 * 
 * @param A 
 * M-by N `Dense` instance to be factorized. 
 * @param sample_size 
 * Determines the the number of random samplings. Usually \p rank + \a p where \a p is an oversampling parameter.
 * @param rank 
 * The rank of \p A. Determines how many columns/rows to be taken (stopping criterion).
 * @param column 
 * If true, a column-ID is computed, otherwise a row-ID is returned.
 * @return std::tuple<Dense, std::vector<int64_t>> 
 * The resulting interpolatory decomposition (ID) given as a `Dense` matrix \p M and a vector \a k
 * containing the corresponding row/column indices of the original matrix \p A.
 * 
 * This function calculates a randomized one-sided interpolatory decomposition (ID) as described in 
 * [Randomized methods for matrix computations](https://arxiv.org/abs/1607.01649v3) by Per-Gunnar Martinsson (2019).
 * 
 * Two flavors of the one-sided ID can be computed:
 * The row-ID spans the row space of \p A and is defined as \f$A\approx XA(I_S,:)\f$.
 * The column-ID spans the column space of \p A and is defined as \f$A\approx A(:,J_S)Z\f$.
 * 
 * Depending on the value of `column`, either \p X or \p Z is returned in \p M and
 * \a k contains the corresponding index set (i.e. \f$I_S\f$ or \f$J_S\f$).
 */
std::tuple<Dense, std::vector<int64_t>> one_sided_rid(
  const Dense& A, int64_t sample_size, int64_t rank, bool column=false);

/**
 * @brief Compute randomized one-sided interpolatory decomposition (ID) of a `Dense` matrix (slow version)
 * 
 * @param A 
 * M-by N `Dense` instance to be factorized. 
 * @param sample_size 
 * Determines the the number of random samplings. Usually \p rank + \a p where \a p is an oversampling parameter.
 * @param rank 
 * The rank of \p A. Determines how many columns/rows to be taken (stopping criterion).
 * @param transA 
 * If true, a row-ID is computed, otherwise a column-ID is returned.
 * @return std::tuple<Dense, std::vector<int64_t>> 
 * The resulting interpolatory decomposition (ID) given as a `Dense` matrix \p M and a vector \a k
 * containing the corresponding row/column indices of the original matrix \p A.
 * 
 * Two flavors of the one-sided ID can be computed:
 * The row-ID spans the row space of \p A and is defined as \f$A\approx XA(I_S,:)\f$.
 * The column-ID spans the column space of \p A and is defined as \f$A\approx A(:,J_S)Z\f$.
 * 
 * Depending on the value of `column`, either \p X or \p Z is returned in \p M and
 * \a k contains the corresponding index set (i.e. \f$I_S\f$ or \f$J_S\f$).
 * 
 * Due to the matrix approximation used, this function is considerably slower than `one_sided_rid()`.
 * The row-ID takes about twice as long while the column-ID is roughly four times slower.
 */
std::tuple<Dense, std::vector<int64_t>> old_one_sided_rid(
  const Dense& A, int64_t sample_size, int64_t rank, bool transA=false);

std::tuple<Dense, Dense, Dense> rid(
  const Dense&, int64_t sample_size, int64_t rank);

std::tuple<Dense, Dense, Dense> rsvd(const Dense&, int64_t sample_size);

std::tuple<Dense, Dense, Dense> old_rsvd(const Dense&, int64_t sample_size);

} // namespace hicma

#endif // hicma_oprations_randomized_factorizations_h
