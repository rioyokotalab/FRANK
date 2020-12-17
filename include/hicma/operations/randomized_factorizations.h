#ifndef hicma_oprations_randomized_factorizations_h
#define hicma_oprations_randomized_factorizations_h

#include <cstdint>
#include <tuple>
#include <vector>


/**
 * @brief General namespace of the HiCMA library
 */
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
 * Depending on the value of \p column, either \p X or \p Z is returned in \p M and
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
 * Depending on the value of \p transA, either \p X or \p Z is returned in \p M and
 * \a k contains the corresponding index set (i.e. \f$I_S\f$ or \f$J_S\f$).
 *
 * Due to the matrix approximation used, this function is considerably slower than `one_sided_rid()`.
 * The row-ID takes about twice as long while the column-ID is roughly four times slower.
 */
std::tuple<Dense, std::vector<int64_t>> old_one_sided_rid(
  const Dense& A, int64_t sample_size, int64_t rank, bool transA=false);

/**
 * @brief Compute a randomized ID (Interpolatory Decomposition) of a a `Dense` matrix.
 *
 * @param A
 * M-by-N `Dense` instance to be factorized.
 * @param sample_size
 * Determines the the number of random samplings. Usually \p rank + \a p where \a p is an oversampling parameter.
 * @param rank
 * The rank of \p A. Determines how many columns/rows to be taken (stopping criterion).
 * @return std::tuple<Dense, Dense, Dense>
 * The resulting double sided interpolatory decomposition (ID) represented by the three `Dense` matrices \p X, \p D and \p Z.
 *
 * This function calculates a randomized double-sided interpolatory decomposition (ID), given by the three `Dense` matrices \p X [M x rank] \p D [rank x rank] and \p Z [rank x N].such that
 * \f$A = XDS\f$ and  \f$D \approx A(I_S,J_S)\f$, spanning both the column and row space of the original matrix \p A.
 * containing the corresponding row/column indices of the original matrix \p A.
 *
 * This function slightly deviates from the description in [Randomized methods for matrix computations](https://arxiv.org/abs/1607.01649v3) by Per-Gunnar Martinsson (2019), since the ID is built
 * from an matrix \p A* which approximates the row and column space of \p A by random sampling.
 */
std::tuple<Dense, Dense, Dense> rid(
  const Dense& A, int64_t sample_size, int64_t rank);

/**
 * @brief Calculates a randomized singular value decomposition (SVD) of a `Dense` matrix.
 *
 * @param A
 * M-by-N `Dense` instance to be factorized.
 * @param sample_size
 * Determines the the number of random samplings. Usually \p rank + \a p where \a p is an oversampling parameter.
 * @return std::tuple<Dense, Dense, Dense>
 * The resulting singular value decompositon represented by the orthonormal matrices U and V and the diagonal matrix D.
 *
 * The ramdomized SVD is calculated according to the algorithm provided in [Randomized methods for matrix computations](https://arxiv.org/abs/1607.01649v3) by Per-Gunnar Martinsson (2019) based on
 * an approximation for the range of A calculated by random sampling.
 * The SVD is calculated from the approximation \f$A \approx QQ^TA\f$, where Q is a matrix with orthonormal columns, approximating the column space of \p A.
 * The SVD is the given by \f$A [M x N] \approx U [M x sample_size] D [sample_size x sample_size] V^T [sample_size x N]\f$.
 */
std::tuple<Dense, Dense, Dense> rsvd(const Dense& A, int64_t sample_size);

/**
 * @brief Calculates a randomized singular value decomposition (SVD) of a `Dense` matrix.
 *
 * @param A
 * M-by-N `Dense` instance to be factorized.
 * @param sample_size
 * Determines the the number of random samplings. Usually \p rank + \a p where \a p is an oversampling parameter.
 * @return std::tuple<Dense, Dense, Dense>
 * The resulting singular value decompositon represented by the orthonormal matrices U and V and the diagonal matrix D.
 *
 * The ramdomized SVD is calculated according to the algorithm provided in [Randomized methods for matrix computations](https://arxiv.org/abs/1607.01649v3) by Per-Gunnar Martinsson (2019) based on
 * an approximation for the range of A calculated by random sampling.
 * The SVD is calculated from the approximation \f$A \approx QQ^TA\f$, where Q is a matrix with orthonormal columns, approximating the column space of \p A.
 * The SVD is the given by \f$A [M x N] \approx U [M x sample_size] D [sample_size x sample_size] V^T [sample_size x N]\f$.
 *
 * This version uses a manual optimization by calculating a QR of the approximation matrix before taking the SVD. However, it seems this optimization technique is already implemented
 * in the corresponding call to BLAS, as no differences in runtime can be observed.
 */
std::tuple<Dense, Dense, Dense> old_rsvd(const Dense& A, int64_t sample_size);

} // namespace hicma

#endif // hicma_oprations_randomized_factorizations_h
