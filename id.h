#ifndef id_h
#define id_h

/* NOTE TO THE WISE
 *
 * You MUST tranpose V that you get from the ID function after the computation is done.
 */

#include <vector>

namespace hicma {
  /*
  void matrix_matrix_mult(
                          const std::vector<double>& A,
                          std::vector<double>& B,
                          std::vector<double>& C,
                          int nrows_a,
                          int ncols_a,
                          int nrows_b,
                          int ncols_b
                          );

  void initialize_random_matrix(std::vector<double>& M, int nrows, int ncols);

  void compute_QR_compact_factorization(
                                        std::vector<double>& Bt,
                                        std::vector<double>& Q,
                                        std::vector<double>& R,
                                        int nrows,
                                        int ncols,
                                        int rank
                                        );

  void QR_factorization_getQ(std::vector<double>& M, std::vector<double>& Q, int nrows, int ncols, int rank);

  void build_diagonal_matrix(std::vector<double>& dvals, int n, std::vector<double>& D);

  void form_svd_product_matrix(
                               std::vector<double>& U,
                               std::vector<double>& S,
                               std::vector<double>& V,
                               std::vector<double>& P,
                               int nrows,
                               int ncols,
                               int rank
                               );

  void calculate_svd(
                     std::vector<double>& U,
                     std::vector<double>& S,
                     std::vector<double>& Vt,
                     std::vector<double>& M,
                     int nrows,
                     int ncols,
                     int rank
                     );
  */
  void randomized_low_rank_svd2(
                                const std::vector<double>& M,
                                int rank,
                                std::vector<double>& U,
                                std::vector<double>& S,
                                std::vector<double>& V,
                                int nrows ,
                                int ncols
                                );

  void transpose(std::vector<double>&  mat, std::vector<double>& mat_t, int nrows, int ncols);
}
#endif
