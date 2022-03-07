#include "hicma/operations/LAPACK.h"
#include "hicma/operations/misc.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/timer.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/yomm2/cute.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <utility>
#include <vector>


namespace hicma
{

std::tuple<Dense, std::vector<int64_t>> geqp3(Matrix& A) {
  return geqp3_omm(A);
}

// Fallback default, abort with error message
define_method(DenseIndexSetPair, geqp3_omm, (Dense& A)) {
  timing::start("DGEQP3");
  // TODO The 0 initial value is important! Otherwise axes are fixed and results
  // can be wrong. See netlib dgeqp3 reference.
  // However, much faster with -1... maybe better starting values exist?
  std::vector<int> jpvt(A.dim[1], 0);
  std::vector<double> tau(std::min(A.dim[0], A.dim[1]), 0);
  LAPACKE_dgeqp3(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1],
    &A, A.stride,
    &jpvt[0], &tau[0]
  );
  // jpvt is 1-based, bad for indexing!
  std::vector<int64_t> column_order(jpvt.size());
  for (size_t i=0; i<jpvt.size(); ++i) column_order[i] = jpvt[i] - 1;
  timing::start("R construction");
  Dense R(A.dim[1], A.dim[1]);
  for(int64_t i=0; i<std::min(A.dim[0], R.dim[0]); i++) {
    for(int64_t j=i; j<R.dim[1]; j++) {
      R(i, j) = A(i, j);
    }
  }timing::stop("R construction");
  timing::stop("DGEQP3");
  return {std::move(R), std::move(column_order)};
}

// Fallback default, abort with error message
define_method(DenseIndexSetPair, geqp3_omm, (Matrix& A)) {
  omm_error_handler("geqp3", {A}, __FILE__, __LINE__);
  std::abort();
}

// Compute truncated rank revealing factorization based on relative threshold
// Modification of LAPACK geqp3 routine
std::tuple<Dense, Dense> truncated_geqp3(const Dense& _A, double eps) {
  // Pointer aliases
  Dense A(_A);
  double* a = &A;
  int m = A.dim[0];
  int n = A.dim[1];
  int lda = A.stride;

  // Initialize variables for pivoted QR
  double tol = LAPACKE_dlamch('e');
  double tol3z = std::sqrt(tol);
  int min_dim = std::min(m, n);
  std::vector<double> tau(min_dim, 0);
  std::vector<double> ipiv(n, 0);
  std::vector<double> cnorm(n, 0);
  std::vector<double> partial_cnorm(n, 0);
  for(int j=0; j<n; j++) {
    ipiv[j] = j;
    cnorm[j] = cblas_dnrm2(m, a + j, lda);
    partial_cnorm[j] = cnorm[j];
  }

  // Begin pivoted QR
  int r = 0;
  double threshold = eps*std::sqrt(norm(A));
  double max_cnorm = *std::max_element(cnorm.begin(), cnorm.end());
  //Handle zero matrix case
  if(max_cnorm <= tol) {
    Dense Q(m, 1); Q(0,0) = 1.0;
    Dense R(1, n);
    return {Q, R};
  }
  while((r < min_dim) && (max_cnorm > threshold)) {
    // Select pivot column and swap
    int k = std::max_element(cnorm.begin() + r, cnorm.end()) - cnorm.begin();
    cblas_dswap(m, a + r, lda, a + k, lda);
    std::swap(cnorm[r], cnorm[k]);
    std::swap(partial_cnorm[r], partial_cnorm[k]);
    std::swap(ipiv[r], ipiv[k]);

    // Generate householder reflector to annihilate A(r+1:m, r)
    double *arr = a + r + (r * lda);
    if(r < (m-1)) {
      // LAPACKE_dlarfg does not work with row major storage so a separate column vector is used
      std::vector<double> ar(m-r, 0);
      cblas_dcopy(m-r, arr, lda, &ar[0], 1);
      LAPACKE_dlarfg(m-r, &ar[0], &ar[1], 1, &tau[r]);
      cblas_dcopy(m-r, &ar[0], 1, arr, lda);
    }
    else {
      LAPACKE_dlarfg(1, arr, arr, 1, &tau[r]);
    }
    
    if(r < (min_dim-1)) {
      // Apply reflector to A(r:m,r+1:n) from left
      double _arr = A(r, r);
      A(r, r) = 1.0;
      // w = A(r:m, r+1:n)^T * v
      std::vector<double> w(n-r-1, 0);
      double *arj = a + r+1 + r * lda;
      cblas_dgemv(CblasRowMajor, CblasTrans,
		  m-r, n-r-1, 1, arj, lda, arr, lda, 0, &w[0], 1);
      // A(r:m,r+1:n) = A(r:m,r+1:n) - tau * v * w^T
      cblas_dger(CblasRowMajor, m-r, n-r-1, -tau[r], arr, lda, &w[0], 1, arj, lda);
      A(r, r) = _arr;
    }
    // Update partial column norm
    for(int j=r+1; j<n; j++) {
      //See LAPACK Working Note 176 (Section 3.2.1) for detail
      if(cnorm[j] != 0.0) {
	double temp = std::fabs(A(r, j)/cnorm[j]);
	temp = std::fmax(0.0, (1-temp)*(1+temp));
	double temp2 = temp * (cnorm[j]/partial_cnorm[j]) * (cnorm[j]/partial_cnorm[j]);
	if(temp2 > tol3z) {
	  cnorm[j] = cnorm[j] * std::sqrt(temp);
	}
	else {
	  if(r < (m-1)) {
	    cnorm[j] = cblas_dnrm2(m-r-1, a+j+(r+1)*lda, lda);
	    partial_cnorm[j] = cnorm[j];
	  }
	  else {
	    cnorm[j] = 0.0;
	    partial_cnorm[j] = 0.0;
	  }
	}
      }
    }
    r++;
    max_cnorm = *std::max_element(cnorm.begin() + r, cnorm.end());
  }
  // Construct truncated Q
  Dense Q(m, r);
  // Copy strictly lower triangular (or trapezoidal) part of A into Q
  for(int i=0; i<Q.dim[0]; i++) {
    for(int j=0; j<std::min(i, r); j++) {
      Q(i, j) = A(i, j);
    }
  }
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], r, &Q, Q.stride, &tau[0]);
  // Construct truncated R
  Dense R(r, n);
  // Copy first r rows of upper triangular part of A into R
  for(int i=0; i<r; i++) {
    for(int j=i; j<n; j++) {
      R(i, j) = A(i, j);
    }
  }
  // Permute columns of R
  std::vector<int> ipivT(ipiv.size(), 0);
  for(size_t i=0; i<ipiv.size(); i++) ipivT[ipiv[i]] = i;
  Dense RP(R);
  for(int i=0; i<R.dim[0]; i++) {
    for(int j=0; j<R.dim[1]; j++) {
      RP(i, j) = R(i, ipivT[j]);
    }
  }
  // Return truncated Q and permuted R
  return {Q, RP};
}

} // namespace hicma
