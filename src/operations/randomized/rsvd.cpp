#include "hicma/operations/randomized_factorizations.h"

#include "hicma/classes/dense.h"
#include "hicma/functions.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"
#include "hicma/util/print.h"
#include "hicma/util/l2_error.h"
#include "hicma/util/timer.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#include <cblas.h>
#endif

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>

void update_norms (const hicma::Dense& A, std::vector<double>& norms, std::vector<double>& exp_norms, int64_t k);


namespace hicma
{

std::tuple<Dense, Dense, Dense> rsvd(const Dense& A, int64_t sample_size) {
  Dense RN(
    random_uniform, std::vector<std::vector<double>>(), A.dim[1], sample_size);
  Dense Y = gemm(A, RN);
  Dense Q(Y.dim[0], Y.dim[1]);
  Dense R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense QtA = gemm(Q, A, 1, true, false);
  Dense Ub, S, V;
  std::tie(Ub, S, V) = svd(QtA);
  // TODO Resizing Ub (and thus U) before this operation might save some time!
  Dense U = gemm(Q, Ub);
  return {std::move(U), std::move(S), std::move(V)};
}
/*
std::tuple<Dense, Dense, Dense> steputv(const Dense& A, int64_t b) {
  Dense RN(
    random_uniform, std::vector<std::vector<double>>(), A.dim[1], b);
  Dense Y = gemm(A, RN);
  Dense Q(Y.dim[0], Y.dim[1]);
  Dense R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense AV = gemm(A, Q);
  Dense AVb(AV.dim[0], b);
  for (int64_t i=0; i<AVb.dim[0]; ++i)
    for (int64_t j=0; j<AVb.dim[1]; ++j)
      AVb(i,j) = AV(i,j);
  Dense U, D, W;
  std::tie(U,D,W) = svd(AVb);
  Dense UtAVb = gemm(U, AVb, 1, true, false);
  print("Dims", W.dim[0]);
  print("Dims", W.dim[1]);
  return {std::move(U), std::move(D), std::move(W)};
}*/

std::tuple<Dense, Dense, Dense> old_rsvd(const Dense& A, int64_t sample_size) {
  Dense RN(
    random_uniform, std::vector<std::vector<double>>(), A.dim[1], sample_size);
  Dense Y = gemm(A, RN);
  Dense Q(Y.dim[0], Y.dim[1]);
  Dense R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense Bt = gemm(A, Q, 1, true, false);
  Dense Qb(A.dim[1], sample_size);
  Dense Rb(sample_size, sample_size);
  qr(Bt, Qb, Rb);
  Dense Ur, S, Vr;
  std::tie(Ur, S, Vr) = svd(Rb);
  // TODO Resizing Ur (and thus U) before this operation might save some time!
  Dense U = gemm(Q, Ur, 1, false, true);
  // TODO Resizing Vr (and thus V) before this operation might save some time!
  Dense V = gemm(Vr, Qb, 1, true, true);
  return {std::move(U), std::move(S), std::move(V)};
}

std::tuple<Dense, Dense, Dense> rsvd_pow(const Dense& A, int64_t sample_size, int64_t q) {
  Dense RN(
    random_uniform, std::vector<std::vector<double>>(), A.dim[1], sample_size);
  Dense Y = gemm(A, RN);
  for (int64_t i=0; i<q; ++i){
    Dense Z = gemm(A, Y, 1, true, false);
    Y = gemm(A, Z);
  }
  Dense Q(Y.dim[0], Y.dim[1]);
  Dense R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense QtA = gemm(Q, A, 1, true, false);
  Dense Ub, S, V;
  std::tie(Ub, S, V) = svd(QtA);
  // TODO Resizing Ub (and thus U) before this operation might save some time!
  Dense U = gemm(Q, Ub);
  return {std::move(U), std::move(S), std::move(V)};
}

std::tuple<Dense, Dense, Dense> rsvd_powOrtho(const Dense& A, int64_t sample_size, int64_t q) {
  Dense RN(
    random_uniform, std::vector<std::vector<double>>(), A.dim[1], sample_size);
  Dense Y = gemm(A, RN);
  Dense Q(Y.dim[0], Y.dim[1]);
  Dense R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense Wo(Y.dim[0], Y.dim[1]);
  for (int64_t i=0; i<q; ++i){
    Dense W = gemm(A, Q, 1, true, false);
    qr(W, Wo, R);
    Y = gemm(A, Wo);
    qr(Y, Q, R);
  }
  Dense QtA = gemm(Q, A, 1, true, false);
  Dense Ub, S, V;
  std::tie(Ub, S, V) = svd(QtA);
  // TODO Resizing Ub (and thus U) before this operation might save some time!
  Dense U = gemm(Q, Ub);
  return {std::move(U), std::move(S), std::move(V)};
}

std::tuple<Dense, Dense, Dense> rsvd_singlePass(const Dense& A, int64_t sample_size) {
  Dense RN1(
    random_uniform, std::vector<std::vector<double>>(), A.dim[1], sample_size);
  Dense RN2(
    random_uniform, std::vector<std::vector<double>>(), A.dim[0], sample_size);
  Dense Y1 = gemm(A, RN1);
  Dense Y2 = gemm(A, RN2, 1, true, false);
  Dense Q1(Y1.dim[0], Y1.dim[1]);
  Dense R1(Y1.dim[1], Y1.dim[1]);
  Dense Q2(Y2.dim[0], Y2.dim[1]);
  Dense R2(Y2.dim[1], Y2.dim[1]);
  //We need a copy of Y2, because the call to QR destroys the values
  Dense Y2_copy(Y2);
  qr(Y1, Q1, R1);
  qr(Y2, Q2, R2);
  //from (RN2'Q1)X = Y2'Q2 aka Ax=B
  Dense RN2tQ1 = gemm(RN2, Q1, 1, true, false);
  Dense Y2tQ2 = gemm(Y2_copy, Q2, 1, true, false);

  //can use either least squares or direct solver
  LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', sample_size, sample_size, sample_size, &RN2tQ1, RN2tQ1.stride, &Y2tQ2, Y2tQ2.stride);
  //std::vector<int> ipiv(Y2tQ2.dim[0]);
  //LAPACKE_dgesv(LAPACK_ROW_MAJOR, RN2tQ1.dim[0], Y2tQ2.dim[1], &RN2tQ1, RN2tQ1.dim[1], ipiv.data(), &Y2tQ2, Y2tQ2.dim[1]);
  Dense Ub, S, Vb;
  std::tie(Ub, S, Vb) = svd(Y2tQ2);
  // TODO Resizing Ub (and thus U) before this operation might save some time!
  Dense U = gemm(Q1, Ub);
  Dense V = gemm(Vb, Q2, 1, false, true);
  return {std::move(U), std::move(S), std::move(V)};
}

std::tuple<Dense, Dense> aca(const Dense& A, int64_t rank) {
  Dense R(A);
  int64_t m = A.dim[0];
  int64_t n = A.dim[1];
  Dense U(m, rank);
  Dense V (rank, n);
  std::vector<int64_t> piv_row(rank+1);
  std::vector<int64_t> piv_col(rank);
  piv_row[0] = 0;
  //Initialization
  piv_col[0] = cblas_idamax(n, &R, 1);
  double gamma = 1/R(0,piv_col[0]);
  for (int64_t j=0; j<n; ++j)
      V(0,j) = R(0,j)*gamma;
  for (int64_t j=0; j<m; ++j)
      U(j,0) = R(j, piv_col[0]);
  piv_row[1] = cblas_idamax(m-1, &R(1,piv_col[0]), n) + 1;

  for (int64_t i=1; i<rank; ++i){
    for (int64_t j=0; j<n; ++j){
      V(i,j) = R(piv_row[i],j);
      bool selected = false;
      for (int64_t k=0; k<i; ++k){
        //printf(" %f ", V(i,j));
        V(i,j) -= U(piv_row[i],k) * V(k,j);
        //printf(" %f ", V(k,j));
        //printf(" %f ", U(piv_row[i],k));
        //printf(" %f ", V(i,j));
        if (j == piv_col[k])
          selected = true;
      }
      //printf("\n Absolute value at %ld is %f: %d\n", j, std::abs(V(i,j)), selected);
      if ((!selected) &&(std::abs(V(i,j)) > std::abs(V(i, piv_col[i]))))
        piv_col[i] = j;
    }
    //printf("\n Piv is at %ld\n", piv_col[i]);
    gamma = 1/V(i, piv_col[i]);
    for (int64_t j=0; j<n; ++j)
      V(i,j) *= gamma;

    for (int64_t j=0; j<m; ++j){
      U(j,i) = R(j,piv_col[i]);
      bool selected = false;
      for (int64_t k=0; k<i; ++k){
        //printf(" %f ", U(j,i));
        U(j,i) -= V(k, piv_col[i]) * U(j,k);
        //printf(" %f ", V(k, piv_col[i]));
        //printf(" %f ", U(j,k));
        //printf(" %f ", U(j,i));
        //printf(" %ld ", piv_row[k]);
        if ((j == piv_row[k]) || (j==piv_row[i]))
          selected = true;
      }
      //printf("\n RO_WAbsolute value at %ld is %f: %d\n", j, std::abs(U(j,i)), selected);
      if ((!selected) && (std::abs(U(j,i)) > std::abs(U(piv_row[i+1], i))))
        piv_row[i+1] = j;
    }
    //printf("\n RowPiv is at %ld\n", piv_row[i+1]);
  }
  /*
    printf("\nSelected columns: ");
    for (int64_t i=0;i<rank;++i)
      printf("%ld ", piv_col[i]);
    printf("\nSelected rows: ");
    for (int64_t i=0;i<=rank;++i)
      printf("%ld ", piv_row[i]);
    printf("\n");
    */

    /*timing::start("FindMax");
    int idx = cblas_idamax(n, &R, 1);
    int x = idx / A.dim[1];
    int y = idx % A.dim[1];
    timing::stop("FindMax");*/
  return {std::move(U), std::move(V)};
}

std::tuple<Dense, Dense> aca_complete(const Dense& A, int64_t rank) {
  Dense R(A);
  Dense U(A.dim[0], rank);
  Dense V (rank, A.dim[1]);
  int64_t n= A.size();
  for (int64_t i=0; i<rank; ++i){
    timing::start("FindMax");
    int idx = cblas_idamax(n, &R, 1);
    int x = idx / A.dim[1];
    int y = idx % A.dim[1];
    timing::stop("FindMax");
    double gamma = 1 / R(x,y);
    timing::start("Copy vectors");
    for (int64_t j=0; j<R.dim[0]; ++j)
      U(j,i) = R(j, y)*gamma;
    for (int64_t j=0; j<R.dim[1]; ++j)
      V(i,j) = R(x, j);
    timing::stop("Copy vectors");
    timing::start("DGEMM");
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, R.dim[0], R.dim[1], 1, -1, &U(0,i), U.dim[1], &V(i,0), V.dim[1], 1, &R, R.dim[1]);
    timing::stop("DGEMM");
  }
  return {std::move(U), std::move(V)};
}

void cpqr_unb(bool pivoting, int64_t cols, int64_t m, int64_t n, Dense& A, int64_t lda, std::vector<int64_t>& pivots, std::vector<double>& tau) {
  std::vector<double> norms(n);
  std::vector<double> exp_norms(n);

  if (pivoting){
    for (int64_t i=0; i<n; ++i)
      exp_norms[i] = norms[i] = LAPACKE_dlange(LAPACK_ROW_MAJOR, 'E', m, 1, &A(0,i), lda);
  }
  //loop until cols columns have been identified
  for (int64_t i=0; i<cols; ++i){
    if (pivoting){
      int64_t idx_max_col = i;
      for (int64_t j=i+1; j<n; ++j)
        if (norms[j] > norms[idx_max_col])
          idx_max_col = j;
      //swap columns
      for (int64_t j=0; j<m; ++j)
        std::swap(A(j,i), A(j, idx_max_col));
      std::swap(norms[i], norms[idx_max_col]);
      std::swap(exp_norms[i], exp_norms[idx_max_col]);
      std::swap(pivots[i], pivots[idx_max_col]);
    }
    /*
    printf("\nA(after pivot):\n");
      for(int64_t i=0; i<M.dim[0]; i++) {
        for(int64_t j=0; j<M.dim[1]; j++) {
          printf("%f ", A(i,j));
        }
        printf("\n");
      }
    */
    //double tau;
    LAPACKE_dlarfg(m-i, &A(i,i), &A(i+1,i), lda, &tau[i]);
    /*printf("\nA(after dlarfg):\n");
      for(int64_t i=0; i<A.dim[0]; i++) {
        for(int64_t j=0; j<A.dim[1]; j++) {
          printf("%f ", A(i,j));
        }
        printf("\n");
      }
    */
    std::vector<double> v(m-i);
    v[0] = 1;
    //printf("\nV: ");
    for (int64_t j=1; j<m-i;++j){
      v[j] = A(i+j,i);
      //printf("%f ", v[j]);
    }
    std::vector<double> work (n-1);
    LAPACKE_dlarfx(LAPACK_ROW_MAJOR, 'L', m-i, n-(i+1), v.data(), tau[i], &A(i, i+1), lda, work.data());
    /*
    printf("\nA(after dlarfx):\n");
    for(int64_t i=0; i<M.dim[0]; i++) {
      for(int64_t j=0; j<M.dim[1]; j++) {
        printf("%f ", A(i,j));
      }
      printf("\n");
    }
    */
    // update partial column norms
    update_norms(A, norms, exp_norms, i);
  }
}

std::tuple<Dense, Dense, Dense> pqr_block(bool pivoting, const Dense& M, int64_t rank) {
  timing::start("CPQR-rank"); 
  Dense A(M);
  /*printf("M:\n");
  for(int64_t i=0; i<M.dim[0]; i++) {
    for(int64_t j=0; j<M.dim[1]; j++) {
      printf("%f ", A(i,j));
    }
    printf("\n");
  }*/
  std::vector<int64_t> pivots(A.dim[1]);
  std::vector<double> tau(rank);
  for (int64_t i=0; i<A.dim[1]; ++i)
    pivots[i] = i;
  timing::start("DGEQP3");
  cpqr_unb(true, rank, A.dim[0], A.dim[1], A, A.dim[0], pivots, tau);
  /*
  int64_t m = A.dim[0];
  int64_t n = A.dim[1];
  std::vector<double> norms(n);
  std::vector<double> exp_norms(n);
  

  if (pivoting){
    for (int64_t i=0; i<n; ++i)
      exp_norms[i] = norms[i] = LAPACKE_dlange(LAPACK_ROW_MAJOR, 'E', m, 1, &A(0,i), m);
  }

  //loop until k columns have been identified
  for (int64_t i=0; i<5; ++i){
    if (pivoting){
      int64_t idx_max_col = i;
      for (int64_t j=i+1; j<n; ++j)
        if (norms[j] > norms[idx_max_col])
          idx_max_col = j;
      //swap columns
      for (int64_t j=0; j<m; ++j)
        std::swap(A(j,i), A(j, idx_max_col));
      std::swap(norms[i], norms[idx_max_col]);
      std::swap(exp_norms[i], exp_norms[idx_max_col]);
      std::swap(pivots[i], pivots[idx_max_col]);
    }
    printf("\nA(after pivot):\n");
  for(int64_t i=0; i<M.dim[0]; i++) {
    for(int64_t j=0; j<M.dim[1]; j++) {
      printf("%f ", A(i,j));
    }
    printf("\n");
  }
    double tau;
    LAPACKE_dlarfg(m-i, &A(i,i), &A(i+1,i), m, &tau);
    printf("\nA(after dlarfg):\n");
  for(int64_t i=0; i<M.dim[0]; i++) {
    for(int64_t j=0; j<M.dim[1]; j++) {
      printf("%f ", A(i,j));
    }
    printf("\n");
  }
    //double diag = A(i,i);
    print("Tau: ", tau);
    std::vector<double> v(m-i);
    v[0] = 1;
    printf("\nV: ");
    for (int64_t j=1; j<m-i;++j){
      v[j] = A(i+j,i);
      printf("%f ", v[j]);
    }
    //A(i,i) = 1;
    std::vector<double> work (n-1);
    LAPACKE_dlarfx(LAPACK_ROW_MAJOR, 'L', m-i, n-(i+1), v.data(), tau, &A(i, i+1), m, work.data());
    printf("\nA(after dlarfx):\n");
  for(int64_t i=0; i<M.dim[0]; i++) {
    for(int64_t j=0; j<M.dim[1]; j++) {
      printf("%f ", A(i,j));
    }
    printf("\n");
  }

    // update partial column norms
    update_norms(A, norms, exp_norms, i);*/
    /*printf("OLD NORMS: ");
    for(int64_t j=0; j<n; j++) 
      printf("%f ", norms[j]);
    double tol3z = std::sqrt(LAPACKE_dlamch('E'));
    for (int64_t j=i+1; j<n; ++j){
      double temp = std::abs(A(i,j)) /norms[j];
      temp = std::max(0.0, (1+temp)*(1-temp));
      double temp5 = norms[j] / norms[j];
      double temp2 = temp * temp5 * temp5;
      if (temp2 <= tol3z){
        printf("\nRECALCULATE NORMS\n");
        // add non-square row check
        norms[j] = norms[i] = LAPACKE_dlange(LAPACK_ROW_MAJOR, 'E', m-j, 1, &A(i,j), m);
      }
      else 
        norms[j] = norms[j] * std::sqrt(temp);
    }
    printf("\nUPDATED NORMS: ");
    for(int64_t j=0; j<n; j++) 
      printf("%f ", norms[j]);*/
  //}
  /*
  printf("PIVOTS: ");
  for(int64_t j=0; j<A.dim[1]; j++) 
    printf("%ld", pivots[j]);

  printf("\nA:\n");
  for(int64_t i=0; i<A.dim[0]; i++) {
    for(int64_t j=0; j<A.dim[1]; j++) {
      printf("%f ", A(i,j));
    }
    printf("\n");
  */
  timing::stop("DGEQP3");
  Dense U(A.dim[0], rank);
  Dense V(rank, A.dim[1]);
  for (int64_t i=0; i<rank; ++i)
    for (int64_t j=0; j<V.dim[1]; ++j)
      V(i, j) = A(i, j);

  timing::start("DORGQR");
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, A.dim[0], rank, rank, &A, A.dim[0], tau.data());
  timing::stop("DORGQR");
  timing::stop("CPQR-rank");
  timing::start("Unnecessary Cleanup"); 
  
  //printf("A:(DORGQR)\n");
  for(int64_t i=0; i<A.dim[0]; i++) {
    for(int64_t j=0; j<rank; j++) {
      U(i,j) = A(i,j);
      //printf("%f ", A(i,j));
    }
    //printf("\n");
  }
 
  Dense Result(M);
  for (int64_t i=0; i<Result.dim[1]; ++i)
    for (int64_t j=0; j<Result.dim[0]; ++j)
      Result(j, i) = M(j, pivots[i]);
  print("Rel. L2 Error", l2_error(Result, gemm(U,V)), false);
  timing::stop("Unnecessary Cleanup");

  timing::start("CPQR-full");
  Dense Q(M);
  std::vector<int> perm(Q.dim[1], 0);
  int64_t k = std::min(Q.dim[0], Q.dim[1]);
  std::vector<double> tau2(k);
  timing::start("DGEQP3");
  LAPACKE_dgeqp3(LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], &Q, Q.dim[1], perm.data(), tau2.data());
  timing::stop("DGEQP3");
  Dense R(k, Q.dim[1]);
  /*
  printf("PERM: ");
  for(int64_t j=0; j<Q.dim[1]; j++) 
    printf("%d", perm[j]);

  printf("\nQ:(DGEQP3)\n");
  for(int64_t i=0; i<Q.dim[0]; i++) {
    for(int64_t j=0; j<Q.dim[1]; j++) {
      printf("%f ", Q(i,j));
      if(j>=i)
        R(i, j) = Q(i, j);
    }
    printf("\n");
  }*/
  Dense U2(Q.dim[0], rank);
  Dense V2(rank, Q.dim[1]);
  for (int64_t i=0; i<rank; ++i)
    for (int64_t j=0; j<V2.dim[1]; ++j)
      V2(i, j) = Q(i, j);

  timing::start("DORGQR");
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], k, &Q, Q.stride, &tau2[0]);
  timing::stop("DORGQR");
  timing::stop("CPQR-full");
  /*printf("Q:(DORGQR)\n");
  for(int64_t i=0; i<Q.dim[0]; i++) {
    for(int64_t j=0; j<Q.dim[1]; j++) {
      printf("%f ", Q(i,j));
    }
    printf("\n");
  }*/
  timing::start("Error Calculation");
  for (int64_t i=0; i<U2.dim[0]; ++i)
    for (int64_t j=0; j<rank; ++j)
      U2(i, j) = Q(i, j);
  Dense Result2(M);
  for (int64_t i=0; i<Result.dim[1]; ++i)
    for (int64_t j=0; j<Result.dim[0]; ++j)
      Result(j, i) = M(j, perm[i]-1);
  print("Rel. L2 Error", l2_error(Result, gemm(U2,V2)), false);
  Dense S(rank, rank);
  timing::stop("Error Calculation");
  return {std::move(U2), std::move(S), std::move(V2)};

}


std::tuple<Dense, Dense, Dense> rrqr(const Dense& M, int64_t rank) {
  double f = 1.05;
  Dense Q(M);
  std::vector<int> perm(Q.dim[1], 0);
  int64_t k = std::min(Q.dim[0], Q.dim[1]);
  std::vector<double> tau(k);
  timing::start("DGEQP3");
  LAPACKE_dgeqp3(LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], &Q, Q.dim[1], perm.data(), tau.data());
  timing::stop("DGEQP3");
  Dense R(k, Q.dim[1]);

  for(int64_t i=0; i<Q.dim[0]; i++) {
    for(int64_t j=0; j<Q.dim[1]; j++) {
      if(j>=i)
        R(i, j) = Q(i, j);
    }
  }
  timing::start("DORGQR");
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], k, &Q, Q.stride, &tau[0]);
  timing::stop("DORGQR");
  if (rank == Q.dim[1])
    throw 42; //placeholder for is_not_low_rank

  timing::start("Initalizing RRQR");
  // make diagonal non negative
  for(int64_t i=0; i<k; i++) {
    if (R(i, i) < 0){
      for (int64_t j=i; j<R.dim[1]; ++j){
        R(i, j) = -R(i, j);
      }
      for (int64_t j=0; j<Q.dim[0]; ++j){
        Q(j, i) = -Q(j, i);
      }
    }
  }
  Dense A(rank, rank);
  int64_t b_cols = R.dim[1]-rank;
  Dense B(rank, b_cols);
  for (int16_t i=0; i<rank; ++i){
    for (int64_t j=0; j<rank; ++j){
      A(i, j) = R(i, j);
    }
    for (int64_t j=0; j<b_cols; ++j)
      B(i, j) = R(i, rank+j);
  }
  LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'U', 'N', rank, &A, rank);
  Dense AB = gemm(A,B);
  
  std::vector<double> gamma(b_cols);
  std::vector<double> omega(rank);
  for (int64_t i=0; i<rank; ++i){
    for (int64_t j=0; j<rank; ++j)
      omega[i] += A(i,j)*A(i,j);
    omega[i] = 1.0/std::sqrt(omega[i]);
  }
  for (int16_t i=0; i<b_cols; ++i){
    for (int64_t j=0; j<R.dim[0]-rank; ++j)
      gamma[i] += R(rank+j, rank+i)*R(rank+j, rank+i);
    gamma[i] = std::sqrt(gamma[i]);
  }
  timing::stop("Initalizing RRQR");

  //main RSSQR Loop
  timing::start("Main RRQR Loop");
  bool finished = false;
  while(!finished){
    int64_t row = -1;
    int64_t col = -1;
    for (int64_t i=0; i<rank && row==-1; ++i)
      for (int64_t j=0; j<b_cols && col==-1; ++j){
        if (AB(i,j)*AB(i,j)+(gamma[j]/omega[i]*gamma[j]/omega[i]) > f*f){
          row = i;
          col = j;
          //TODO remove
          //finished=true;
        }}
    if (row == -1 || col ==-1){
      finished = true;
      break;
    }
    //start interchanging
    // interchange rank and rank+col columns
    if (col > 0){
      std::swap(gamma[0], gamma[col]);
      std::swap(perm[rank], perm[rank+col]);
      for (int64_t i=0; i<AB.dim[0]; ++i)
        std::swap(AB(i, 0), AB(i, col));
      for (int64_t i=0; i<R.dim[0]; ++i)
        std::swap(R(i, rank), R(i, rank+col));
    }

    // interchanging the i and rank columns
    if (row < rank-1){
      std::rotate(omega.begin()+row, omega.begin()+row+1, omega.begin()+rank);
      std::rotate(perm.begin()+row, perm.begin()+row+1, perm.begin()+rank);
      for (int64_t i=0; i<AB.dim[1]; ++i)
        for(int64_t j=row; j<rank-1; ++j)
          std::swap(AB(j, i), AB(j+1, i));
      for (int64_t i=0; i<R.dim[0]; ++i)
        for(int64_t j=row; j<rank-1; ++j)
          std::swap(R(i, j), R(i, j+1));
      // givens rotation for triangulation of R(1:k, 1:k)
      for (int64_t i=row; i<rank-1; ++i){
        double a = R(i, i);
        double b = R(i+1, i);
        double c, s;
        cblas_drotg(&a, &b, &c, &s);
        //guarantee R(i,i) ends up positive
        if (c*R(i, i)+s*R(i+1, i)<0){
          c = -c;
          s = -s;
        }
        cblas_drot(R.dim[0], &R(i,0), 1, &R(i+1, 0), 1, c, s);
        cblas_drot(Q.dim[1], &Q(0,i), Q.dim[1], &Q(0, i+1), Q.dim[1], c, s);
      }
      // assert R(rank-1, rank-1) is positive
      if (R(rank-1, rank-1) < 0){
        for (int64_t i=0; i<R.dim[1]; ++i)
          R(rank-1, i) = -R(rank-1, i);
        for (int64_t i=0; i<Q.dim[0]; ++i)
          Q(i, rank-1) = -Q(i, rank-1);
      }
    }

    // zeroing out the below-diag of the k+1 columns
    if (rank-1 < R.dim[0]){
      for (int64_t i=rank+1; i<R.dim[0]; ++i){
        double a = R(rank, rank);
        double b = R(i, rank);
        double c, s;
        cblas_drotg(&a, &b, &c, &s);
        //guarantee R(i,i) ends up positive
        if (c*R(rank, rank)+s*R(i, rank)<0){
          c = -c;
          s = -s;
        }
        cblas_drot(R.dim[0], &R(rank,0), 1, &R(i, 0), 1, c, s);
        cblas_drot(Q.dim[1], &Q(0,rank), Q.dim[1], &Q(0, i), Q.dim[1], c, s);
      }
    }
    
    // interchanging the k and k+1 columns
    std::swap(perm[rank-1], perm[rank]);
    double ga = R(rank-1, rank-1);
    double mu = R(rank-1, rank)/ga;
    double nu = 0;
    if (rank-1 < R.dim[0])
      nu = R(rank, rank)/ga;
    double rho = std::sqrt(mu*mu+nu*nu);
    double ga_bar = ga*rho;
    std::vector<double> c1T(R.dim[1]-(rank+1));
    std::vector<double> c2T(R.dim[1]-(rank+1));
    for (int64_t i=0; i<R.dim[1]-(rank+1); ++i)
        c1T[i] = R(rank-1, i+rank+1);
    if (rank < R.dim[0])
      for (int64_t i=0; i<R.dim[1]-(rank+1); ++i)
        c2T[i] = R(rank, i+rank+1);

    // immediately write c1T_bar to R
    for (int64_t i=0; i<R.dim[1]-(rank+1); ++i)
      R(rank-1, i+rank+1) = (mu*c1T[i]+nu*c2T[i])/rho;
    if (rank < R.dim[0])
      // immediately write c2T_bar to R
      for (int64_t i=0; i<R.dim[1]-(rank+1); ++i)
        R(rank, i+rank+1) = (nu*c1T[i]-mu*c2T[i])/rho;
    
    std::vector<double> u(rank-1);
    for (int64_t i=0; i<rank-1; ++i)
      u[i] = R(i, rank-1);
    // swap b1 and b2
    for (int64_t i=0; i<rank-1; ++i)
      std::swap(R(i, rank-1), R(i, rank));
    R(rank-1, rank-1) = ga_bar;
    R(rank-1, rank) = ga*mu/rho;
    R(rank, rank) = ga*nu/rho;
    
    // update AB
    LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', rank-1, 1, &R, R.dim[1], u.data(), 1);
    std::vector<double> u1(rank-1);
    for (int64_t i=0; i<rank-1; ++i){
      u1[i] = AB(i, 0);
      AB(i, 0) = (nu*nu*u[i]-mu*u1[i])/(rho*rho);
    }
    AB(rank-1, 0) = mu/(rho*rho);
    for (int64_t i=1; i<AB.dim[1]; ++i){
      AB(rank-1, i) = R(rank-1, i+rank)/ga_bar;
      for (int64_t j=0; j<rank-1; ++j)
        AB(j, i) = AB(j, i)+(nu*u[j]*R(rank, i+rank)-u1[j]*R(rank-1, i+rank))/ga_bar;
    }

    //update gamma
    gamma[0] = ga*nu/rho;
    for (size_t i=1; i<gamma.size(); ++i)
      gamma[i] = std::sqrt(gamma[i]*gamma[i]+R(rank, i+rank)*R(rank, i+rank) - c2T[i-1]*c2T[i-1]);

    //update omega
    omega[rank-1] = ga_bar;
    for (int64_t i=0; i<rank-1; ++i)
      omega[i] = 1/std::sqrt(1.0/(omega[i]*omega[i])+(u1[i]+mu*u[i])*(u1[i]+mu*u[i])/(ga_bar*ga_bar)-u[i]*u[i]/(ga*ga));

    //Eliminate new R(k+1, k) by orthogonal transformation
    if (rank-1 < R.dim[0]){
      for (int64_t i=0; i<Q.dim[0]; ++i){
        double tmp = Q(i, rank-1);
        Q(i, rank-1) = Q(i, rank-1)*mu/rho + Q(i, rank)*nu/rho;
        Q(i, rank) = tmp*nu/rho + Q(i, rank)*(-mu/rho);
      }
    }
  }
  timing::stop("Main RRQR Loop");
  
  timing::start("Unnecessary Cleanup");
  Dense U(Q.dim[0], rank);
  Dense S(rank, rank);
  Dense V(rank, R.dim[1]);
  for (int64_t i=0; i<U.dim[0]; ++i)
    for (int64_t j=0; j<rank; ++j)
      U(i, j) = Q(i, j);
  for (int64_t i=0; i<rank; ++i)
      S(i, i) = 1.0;
  for (int64_t i=0; i<rank; ++i)
    for (int64_t j=0; j<V.dim[1]; ++j)
      V(i, j) = R(i, j);
  Dense Result(M);
  for (int64_t i=0; i<Result.dim[1]; ++i)
    for (int64_t j=0; j<Result.dim[0]; ++j)
      Result(j, i) = M(j, perm[i]-1);
  print("Rel. L2 Error", l2_error(Result, gemm(U,V)), false);
  timing::stop("Unnecessary Cleanup");
  return {std::move(U), std::move(S), std::move(V)};
}

std::tuple<Dense, Dense, Dense> rrlu(const Dense& A, int64_t rank) {
  double u = 1.05;

  // Algorithm 2 - create a rectangular submatrix having a local u-maximum volume
  Dense AtA = gemm(A, A, 1, true, false);
  print(AtA);

  // uncomplete LU of AtA
  std::vector<int64_t> perm_row(AtA.dim[0]);
  std::vector<int64_t> perm_col(AtA.dim[1]);
  for (int64_t i=0; i<AtA.dim[0]; ++i)
    perm_row[i] = perm_col[i] = i;
  for (int64_t k=0; k<rank; k++){
    int64_t p=0;
    int64_t row;
    int64_t col;
    //look for max pivot element
    for (int64_t i=k; i<AtA.dim[0]; ++i){
      for(int64_t j=k; j<AtA.dim[1]; ++j){
        if (std::abs(AtA(i,j)) > p){
	        p = std::abs(AtA(i,j));
	        row = i;
          col = j;
        }
      }
    }
    if (p==0)
      break; //No more work to be done
    std::swap(perm_row[k], perm_row[row]);
    std::swap(perm_col[k], perm_col[col]);
    for (int64_t i=0; i<AtA.dim[1]; ++i)
      std::swap(AtA(k,i), AtA(row,i));
    for (int64_t i=0; i<AtA.dim[0]; ++i)
      std::swap(AtA(i,k), AtA(i,col));
    //in-place LU
    for (int64_t i=k+1; i<AtA.dim[0]; ++i){
      AtA(i,k)=AtA(i,k)/AtA(k,k);
      for (int64_t j=k+1; j<AtA.dim[1]; ++j)
       AtA(i,j)=AtA(i,j)-AtA(i,k)*AtA(k,j); 
    }    
  }
 
  int64_t k = rank-1; //rank column
  int64_t j = rank-2; // rank-1th column
  for (int64_t i=0; i<j; ++i){
    std::swap(AtA(i,j), AtA(i,k));
    std::swap(AtA(j,i), AtA(k,i));

  }
  //TODO verify if this needs to be in the final version
  std::swap(perm_row[j], perm_row[k]);
  std::swap(perm_col[j], perm_col[k]);

  double hi = AtA(j,k);
  double h_sqr = AtA(j,j);
  double j_sqr = AtA(k,k);
  double i_sqr = AtA(k,j)*hi;
  double alpha = i_sqr + j_sqr;
  AtA(j,j) = alpha;
  AtA(j,k) = hi;
  AtA(k,j) = hi/alpha;
  AtA(k,k) = h_sqr*j_sqr/alpha;

  //provoke trigger TODO remove
  AtA(9,9)= 7000;
  Dense L(AtA.dim[0], AtA.dim[1]);
  Dense U(AtA.dim[0], AtA.dim[1]);
  for (int64_t i=0; i<AtA.dim[1]; ++i)
    L(i,i) = 1;
  for (int64_t i=0; i<AtA.dim[0]; ++i)
    for (int64_t j=0; j<AtA.dim[1]; ++j)
      if (i<rank)
        if (j<rank)
          if (i>j)
            L(i,j) = AtA(i,j);
          else
            U(i,j) = AtA(i,j);
        else
          U(i,j) = AtA(i,j);
      else
        if (j<rank)
          L(i,j) = AtA(i,j);
        else
          U(i,j) = AtA(i,j);
  print(U);

  //compare L(i,k:n)U(k:n,i) for i=k..n (i.e. the diagonal values)
  int64_t l = k;
  for (int64_t i=rank; i<AtA.dim[1]; ++i)
    if (AtA(i,i)>AtA(l,l))
      l = i;
  if (u*u*AtA(k,k)>=AtA(l,l))
    j -= 1;
  else {
    //exchange k and l
    if (l>rank)
      for (int64_t i=0; i<A.dim[0]; ++i){
        std::swap(U(i, rank), U(i, l));
        std::swap(L(rank, i), L(l, i));
      }
    for (int64_t i=0; i<A.dim[0]; ++i){
      std::swap(U(i, rank), U(i, k));
      std::swap(L(rank, i), L(k, i));
    }
    j = rank-2;
  }
  Dense In(AtA.dim[0], AtA.dim[1]);
  for (int64_t i=0; i<AtA.dim[0]; ++i)
    In(i,i)=1;
  print(U);
  print(gemm(In,U));
  for (int64_t i=rank; i<AtA.dim[0]; ++i)
    In(i,k)-= U(i,k)/U(k,k);
  print(gemm(In,U));
  print(L);
  print(gemm(L, In, 1, false, true));
  /*
  //Dense L(AtA.dim[0], AtA.dim[1]);
  //Dense U(AtA.dim[0], AtA.dim[1]);
  for (int64_t i=0; i<AtA.dim[1]; ++i)
    L(i,i) = 1;
  for (int64_t i=0; i<AtA.dim[0]; ++i)
    for (int64_t j=0; j<AtA.dim[1]; ++j)
      if (i<rank)
        if (j<rank)
          if (i>j)
            L(i,j) = AtA(i,j);
          else
            U(i,j) = AtA(i,j);
        else
          U(i,j) = AtA(i,j);
      else
        if (j<rank)
          L(i,j) = AtA(i,j);
        else
          U(i,j) = AtA(i,j);
        
  print(L);
  print(U);
  print(gemm(L,U));
  Dense P(AtA.dim[0], AtA.dim[1]);
  Dense Q(AtA.dim[0], AtA.dim[1]);
  for (int64_t i=0; i<AtA.dim[0]; ++i){
      P(i, perm_col[i]) = 1;
      Q(perm_row[i], i) = 1;
  }*/
  //print(P);
  //print(Q);
  //Dense Atest = gemm(A, A, 1, true, false);
  //print(Atest);
  //print(gemm(gemm(P, Atest), Q));



            


  /*std::vector<int> perm(Q.dim[1], 0);
  int64_t k = std::min(Q.dim[0], Q.dim[1]);
  std::vector<double> tau(k);
  timing::start("DGEQP3");
  LAPACKE_dgeqp3(LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], &Q, Q.dim[1], perm.data(), tau.data());
  timing::stop("DGEQP3");
  Dense R(k, Q.dim[1]);

  for(int64_t i=0; i<Q.dim[0]; i++) {
    for(int64_t j=0; j<Q.dim[1]; j++) {
      if(j>=i)
        R(i, j) = Q(i, j);
    }
  }
  timing::start("DORGQR");
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], k, &Q, Q.stride, &tau[0]);
  timing::stop("DORGQR");
  if (rank == Q.dim[1])
    throw 42; //placeholder for is_not_low_rank

  timing::start("Initalizing RRQR");
  // make diagonal non negative
  for(int64_t i=0; i<k; i++) {
    if (R(i, i) < 0){
      for (int64_t j=i; j<R.dim[1]; ++j){
        R(i, j) = -R(i, j);
      }
      for (int64_t j=0; j<Q.dim[0]; ++j){
        Q(j, i) = -Q(j, i);
      }
    }
  }
  Dense A(rank, rank);
  int64_t b_cols = R.dim[1]-rank;
  Dense B(rank, b_cols);
  for (int16_t i=0; i<rank; ++i){
    for (int64_t j=0; j<rank; ++j){
      A(i, j) = R(i, j);
    }
    for (int64_t j=0; j<b_cols; ++j)
      B(i, j) = R(i, rank+j);
  }
  LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'U', 'N', rank, &A, rank);
  Dense AB = gemm(A,B);
  
  std::vector<double> gamma(b_cols);
  std::vector<double> omega(rank);
  for (int64_t i=0; i<rank; ++i){
    for (int64_t j=0; j<rank; ++j)
      omega[i] += A(i,j)*A(i,j);
    omega[i] = 1.0/std::sqrt(omega[i]);
  }
  for (int16_t i=0; i<b_cols; ++i){
    for (int64_t j=0; j<R.dim[0]-rank; ++j)
      gamma[i] += R(rank+j, rank+i)*R(rank+j, rank+i);
    gamma[i] = std::sqrt(gamma[i]);
  }
  timing::stop("Initalizing RRQR");

  //main RSSQR Loop
  timing::start("Main RRQR Loop");
  bool finished = false;
  while(!finished){
    int64_t row = -1;
    int64_t col = -1;
    for (int64_t i=0; i<rank && row==-1; ++i)
      for (int64_t j=0; j<b_cols && col==-1; ++j){
        if (AB(i,j)*AB(i,j)+(gamma[j]/omega[i]*gamma[j]/omega[i]) > f*f){
          row = i;
          col = j;
          //TODO remove
          //finished=true;
        }}
    if (row == -1 || col ==-1){
      finished = true;
      break;
    }
    //start interchanging
    // interchange rank and rank+col columns
    if (col > 0){
      std::swap(gamma[0], gamma[col]);
      std::swap(perm[rank], perm[rank+col]);
      for (int64_t i=0; i<AB.dim[0]; ++i)
        std::swap(AB(i, 0), AB(i, col));
      for (int64_t i=0; i<R.dim[0]; ++i)
        std::swap(R(i, rank), R(i, rank+col));
    }

    // interchanging the i and rank columns
    if (row < rank-1){
      std::rotate(omega.begin()+row, omega.begin()+row+1, omega.begin()+rank);
      std::rotate(perm.begin()+row, perm.begin()+row+1, perm.begin()+rank);
      for (int64_t i=0; i<AB.dim[1]; ++i)
        for(int64_t j=row; j<rank-1; ++j)
          std::swap(AB(j, i), AB(j+1, i));
      for (int64_t i=0; i<R.dim[0]; ++i)
        for(int64_t j=row; j<rank-1; ++j)
          std::swap(R(i, j), R(i, j+1));
      // givens rotation for triangulation of R(1:k, 1:k)
      for (int64_t i=row; i<rank-1; ++i){
        double a = R(i, i);
        double b = R(i+1, i);
        double c, s;
        cblas_drotg(&a, &b, &c, &s);
        //guarantee R(i,i) ends up positive
        if (c*R(i, i)+s*R(i+1, i)<0){
          c = -c;
          s = -s;
        }
        cblas_drot(R.dim[0], &R(i,0), 1, &R(i+1, 0), 1, c, s);
        cblas_drot(Q.dim[1], &Q(0,i), Q.dim[1], &Q(0, i+1), Q.dim[1], c, s);
      }
      // assert R(rank-1, rank-1) is positive
      if (R(rank-1, rank-1) < 0){
        for (int64_t i=0; i<R.dim[1]; ++i)
          R(rank-1, i) = -R(rank-1, i);
        for (int64_t i=0; i<Q.dim[0]; ++i)
          Q(i, rank-1) = -Q(i, rank-1);
      }
    }

    // zeroing out the below-diag of the k+1 columns
    if (rank-1 < R.dim[0]){
      for (int64_t i=rank+1; i<R.dim[0]; ++i){
        double a = R(rank, rank);
        double b = R(i, rank);
        double c, s;
        cblas_drotg(&a, &b, &c, &s);
        //guarantee R(i,i) ends up positive
        if (c*R(rank, rank)+s*R(i, rank)<0){
          c = -c;
          s = -s;
        }
        cblas_drot(R.dim[0], &R(rank,0), 1, &R(i, 0), 1, c, s);
        cblas_drot(Q.dim[1], &Q(0,rank), Q.dim[1], &Q(0, i), Q.dim[1], c, s);
      }
    }
    
    // interchanging the k and k+1 columns
    std::swap(perm[rank-1], perm[rank]);
    double ga = R(rank-1, rank-1);
    double mu = R(rank-1, rank)/ga;
    double nu = 0;
    if (rank-1 < R.dim[0])
      nu = R(rank, rank)/ga;
    double rho = std::sqrt(mu*mu+nu*nu);
    double ga_bar = ga*rho;
    std::vector<double> c1T(R.dim[1]-(rank+1));
    std::vector<double> c2T(R.dim[1]-(rank+1));
    for (int64_t i=0; i<R.dim[1]-(rank+1); ++i)
        c1T[i] = R(rank-1, i+rank+1);
    if (rank < R.dim[0])
      for (int64_t i=0; i<R.dim[1]-(rank+1); ++i)
        c2T[i] = R(rank, i+rank+1);

    // immediately write c1T_bar to R
    for (int64_t i=0; i<R.dim[1]-(rank+1); ++i)
      R(rank-1, i+rank+1) = (mu*c1T[i]+nu*c2T[i])/rho;
    if (rank < R.dim[0])
      // immediately write c2T_bar to R
      for (int64_t i=0; i<R.dim[1]-(rank+1); ++i)
        R(rank, i+rank+1) = (nu*c1T[i]-mu*c2T[i])/rho;
    
    std::vector<double> u(rank-1);
    for (int64_t i=0; i<rank-1; ++i)
      u[i] = R(i, rank-1);
    // swap b1 and b2
    for (int64_t i=0; i<rank-1; ++i)
      std::swap(R(i, rank-1), R(i, rank));
    R(rank-1, rank-1) = ga_bar;
    R(rank-1, rank) = ga*mu/rho;
    R(rank, rank) = ga*nu/rho;
    
    // update AB
    LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', rank-1, 1, &R, R.dim[1], u.data(), 1);
    std::vector<double> u1(rank-1);
    for (int64_t i=0; i<rank-1; ++i){
      u1[i] = AB(i, 0);
      AB(i, 0) = (nu*nu*u[i]-mu*u1[i])/(rho*rho);
    }
    AB(rank-1, 0) = mu/(rho*rho);
    for (int64_t i=1; i<AB.dim[1]; ++i){
      AB(rank-1, i) = R(rank-1, i+rank)/ga_bar;
      for (int64_t j=0; j<rank-1; ++j)
        AB(j, i) = AB(j, i)+(nu*u[j]*R(rank, i+rank)-u1[j]*R(rank-1, i+rank))/ga_bar;
    }

    //update gamma
    gamma[0] = ga*nu/rho;
    for (size_t i=1; i<gamma.size(); ++i)
      gamma[i] = std::sqrt(gamma[i]*gamma[i]+R(rank, i+rank)*R(rank, i+rank) - c2T[i-1]*c2T[i-1]);

    //update omega
    omega[rank-1] = ga_bar;
    for (int64_t i=0; i<rank-1; ++i)
      omega[i] = 1/std::sqrt(1.0/(omega[i]*omega[i])+(u1[i]+mu*u[i])*(u1[i]+mu*u[i])/(ga_bar*ga_bar)-u[i]*u[i]/(ga*ga));

    //Eliminate new R(k+1, k) by orthogonal transformation
    if (rank-1 < R.dim[0]){
      for (int64_t i=0; i<Q.dim[0]; ++i){
        double tmp = Q(i, rank-1);
        Q(i, rank-1) = Q(i, rank-1)*mu/rho + Q(i, rank)*nu/rho;
        Q(i, rank) = tmp*nu/rho + Q(i, rank)*(-mu/rho);
      }
    }
  }
  timing::stop("Main RRQR Loop");

  timing::start("Unnecessary Cleanup");
  Dense U(Q.dim[0], rank);
  Dense S(rank, rank);
  Dense V(rank, R.dim[1]);
  for (int64_t i=0; i<U.dim[0]; ++i)
    for (int64_t j=0; j<rank; ++j)
      U(i, j) = Q(i, j);
  for (int64_t i=0; i<rank; ++i)
      S(i, i) = 1.0;
  for (int64_t i=0; i<rank; ++i)
    for (int64_t j=0; j<V.dim[1]; ++j)
      V(i, j) = R(i, j);
  Dense Result(M);
  for (int64_t i=0; i<Result.dim[1]; ++i)
    for (int64_t j=0; j<Result.dim[0]; ++j)
      Result(j, i) = M(j, perm[i]-1);
  print("Rel. L2 Error", l2_error(Result, gemm(U,V)), false);
  timing::stop("Unnecessary Cleanup");
  return {std::move(U), std::move(S), std::move(V)};*/
}

} // namespace hicma

void update_norms (const hicma::Dense& A, std::vector<double>& norms, std::vector<double>& exp_norms, int64_t k){
  /*printf("OLD NORMS: ");
  for(int64_t j=0; j<n; j++) 
    printf("%f ", norms[j]);*/
  double tol3z = std::sqrt(LAPACKE_dlamch('E'));
  for (size_t j=k+1; j<norms.size(); ++j){
    double temp = std::abs(A(k,j)) / norms[j];
    temp = std::max(0.0, (1+temp)*(1-temp));
    double temp5 = norms[j] / exp_norms[j];
    double temp2 = temp * temp5 * temp5;
    if (temp2 <= tol3z){
      //printf("\nRECALCULATE NORMS %ld", k);
      //printf("\nCOLUMN %ld", j);
      norms[j] = exp_norms[j] = LAPACKE_dlange(LAPACK_ROW_MAJOR, 'E', A.dim[0]-(k+1), 1, &A(k+1,j), A.dim[0]);
    }
    else 
      norms[j] = norms[j] * std::sqrt(temp);
  }
  /*printf("\nUPDATED NORMS: ");
  for(int64_t j=0; j<n; j++) 
    printf("%f ", norms[j]);
  */
}
