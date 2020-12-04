#include "hicma/hicma.h"
#include "hicma/operations/NoFLA_HQRRP_WY_blk_var4.h"
#include <lapacke.h>

#include <cstdint>
#include <tuple>
#include <vector>
#include <cmath>


using namespace hicma;

int main() {
  for (int i=100; i<=5000; i+=100){print("\nRUN with k=",i);
  hicma::initialize();
  int64_t N = i;
  int64_t rank = 16;
  /*Dense D(5,5);
  D(0,0)=2;
  D(0,1)=3;
  D(0,2)=7;
  D(0,3)=4;
  D(0,4)=5;
  D(1,0)=7;
  D(1,1)=10;
  D(1,2)=4;
  D(1,3)=9;
  D(1,4)=7;
  D(2,0)=5;
  D(2,1)=3;
  D(2,2)=5;
  D(2,3)=3;
  D(2,4)=1;
  D(3,0)=4;
  D(3,1)=3;
  D(3,2)=5;
  D(3,3)=2;
  D(3,4)=10;
  D(4,0)=4;
  D(4,1)=9;
  D(4,2)=7;
  D(4,3)=1;
  D(4,4)=5;*/
  /*Dense D(N, N);
  D(0,0)=18;
  D(0,1)=6;
  D(0,2)=37;
  D(0,3)=37;
  D(0,4)=32;
  D(0,5)=30;
  D(0,6)=16;
  D(0,7)=6;;
  D(0,8)=14;
  D(0,9)=20;
  D(1,0)=4;
  D(1,1)=11;
  D(1,2)=1;
  D(1,3)=22;
  D(1,4)=0;
  D(1,5)=17;
  D(1,6)=21;
  D(1,7)=3;
  D(1,8)=22;
  D(1,9)=37;
  D(2,0)=4;
  D(2,1)=23;
  D(2,2)=26;
  D(2,3)=0;
  D(2,4)=15;
  D(2,5)=2;
  D(2,6)=3;
  D(2,7)=4;
  D(2,8)=11;
  D(2,9)=30;
  D(3,0)=5;
  D(3,1)=16;
  D(3,2)=29;
  D(3,3)=21;
  D(3,4)=34;
  D(3,5)=18;
  D(3,6)=8;
  D(3,7)=19;
  D(3,8)=34;
  D(3,9)=1;
  D(4,0)=19;
  D(4,1)=24;
  D(4,2)=0;
  D(4,3)=26;
  D(4,4)=7;
  D(4,5)=3;
  D(4,6)=29;
  D(4,7)=29;
  D(4,8)=14;
  D(4,9)=11;
  D(5,0)=23;
  D(5,1)=22;
  D(5,2)=23;
  D(5,3)=21;
  D(5,4)=30;
  D(5,5)=38;
  D(5,6)=13;
  D(5,7)=9;
  D(5,8)=2;
  D(5,9)=4;
  D(6,0)=17;
  D(6,1)=0;
  D(6,2)=7;
  D(6,3)=3;
  D(6,4)=3;
  D(6,5)=18;
  D(6,6)=19;
  D(6,7)=15;
  D(6,8)=38;
  D(6,9)=39;
  D(7,0)=25;
  D(7,1)=8;
  D(7,2)=31;
  D(7,3)=20;
  D(7,4)=27;
  D(7,5)=28;
  D(7,6)=23;
  D(7,7)=40;
  D(7,8)=7;
  D(7,9)=28;
  D(8,0)=2;
  D(8,1)=30;
  D(8,2)=23;
  D(8,3)=16;
  D(8,4)=32;
  D(8,5)=38;
  D(8,6)=9;
  D(8,7)=5;
  D(8,8)=35;
  D(8,9)=35;
  D(9,0)=18;
  D(9,1)=15;
  D(9,2)=39;
  D(9,3)=11;
  D(9,4)=19;
  D(9,5)=37;
  D(9,6)=2;
  D(9,7)=38;
  D(9,8)=14;
  D(9,9)=37;*/
  timing::start("Init matrix");
  std::vector<std::vector<double>> randx{get_sorted_random_vector(2*N)};
  Dense D(laplacend, randx, N, N, 0, N);
  /*
  std::vector<double> d(N, 0.0); //Singular values to be used
  char dist = 'U';
  std::vector<int> iseed{ 1, 23, 456, 789 };
  char sym = 'N'; //Generate symmetric or non-symmetric matrix
  double dmax = 1.0;
  int64_t kl = N-1;
  int64_t ku = N-1;
  char pack = 'N';
  int64_t mode = 1; //See docs
  Dense D(N, N);
  double conditionNumber = 16.0;
  latms(dist, iseed, sym, d, mode, conditionNumber, dmax, kl, ku, pack, D);
  */

  timing::stopAndPrint("Init matrix");
  /*
  print("PQR");
  timing::start("PQR");
  LowRank PQR(D, rank, pqr);
  timing::stopAndPrint("PQR", 2);
  //print("Rel. L2 Error", l2_error(D, QR), false);
  */
  /*Dense U, S, V;
  timing::start("SVD");
  Dense DC(D);
  std::tie(U, S, V) = svd(DC);
  timing::stopAndPrint("SVD", 2);
  print("Rel. L2 Error", l2_error(D, gemm(gemm(U, S), V)), false);

  print("RSVD");
  timing::start("Randomized SVD");
  LowRank LR(D, rank);
  timing::stopAndPrint("Randomized SVD", 2);
  print("Rel. L2 Error", l2_error(D, LR), false);*/
  /*
  print("RSVD - Power Iteration");
  timing::start("Randomized SVD pow");
  LowRank LRpow(D, rank, powIt);
  timing::stopAndPrint("Randomized SVD pow", 2);
  print("Rel. L2 Error", l2_error(D, LRpow), false);

  print("RSVD - Orthonormalized Power Iteration");
  timing::start("Randomized SVD powOrtho");
  LowRank LRpowOrtho(D, rank, powOrtho);
  timing::stopAndPrint("Randomized SVD powOrtho", 2);
  print("Rel. L2 Error", l2_error(D, LRpowOrtho), false);
  
  print("RSVD - Single Pass");
  timing::start("Randomized SVD singlePass");
  LowRank LRsinglePass(D, rank, singlePass);
  timing::stopAndPrint("Randomized SVD singlePass", 2);
  print("Rel. L2 Error", l2_error(D, LRsinglePass), false);
  */
  /*print("RRQR");
  timing::start("RRQR");
  LowRank QR(D, rank, rankqr);
  timing::stopAndPrint("RRQR", 2);
  print("Rel. L2 Error", l2_error(D, QR), false);*/
  
  /* print(D);
   transpose(D);
   print(D);
  Dense ID;
  std::vector<int64_t> perm;
  std::tie(ID, perm) = one_sided_id(D, rank);
  print(ID);
  for (size_t i=0; i<perm.size();++i)
  printf("%dl\n", perm[i]);*/

  /*print("ID");
  //Dense U, S, V;
  timing::start("ID");
  Dense Dwork(D);
  std::tie(U, S, V) = id(Dwork, rank);
  timing::stopAndPrint("ID", 2);
  Dense test = gemm(gemm(U, S), V);
  print("Rel. L2 Error", l2_error(D, test), false);
  
  print("RID");
  timing::start("Randomized ID");
  std::tie(U, S, V) = rid(D, rank+5, rank);
  timing::stopAndPrint("Randomized ID", 2);
  test = gemm(gemm(U, S), V);
  print("Rel. L2 Error", l2_error(D, test), false);
  
  print("RID - one-sided");
  std::vector<int64_t> selected_cols;
  timing::start("one-sided RID");
  std::tie(V, selected_cols) = one_sided_rid_new(D, rank+5, rank, 0, false);
  timing::stopAndPrint("one-sided RID", 2);
  //Dense P(D.dim[1], rank);
  //for (int64_t i=0; i<rank; ++i)
  //  P(selected_cols[i], i) = 1;
  //test = gemm(gemm(D, P), V, 1, false, false);
  Dense P(rank, D.dim[0]);
  for (int64_t i=0; i<rank; ++i)
    P(i, selected_cols[i]) = 1;
  test = gemm(V, gemm(P, D), 1, true, false);
  print("Rel. L2 Error", l2_error(D, test), false);*/
  
  print("adaptive cross approximation ACA");
  Dense U,V;
  timing::start("ACA");
  std::tie(U, V) = aca(D, rank);
  timing::stopAndPrint("ACA", 2);
  Dense test = gemm(U,V);
  //print(U);
  //print(V);
  print("Rel. L2 Error", l2_error(D, test), false);

  print("adaptive cross approximation ACA - old");
  //std::vector<int64_t> selected_cols;
  timing::start("ACA2");
  std::tie(U, V) = aca_complete(D, rank);
  timing::stopAndPrint("ACA2", 2);
  test = gemm(U,V);
  print("Rel. L2 Error", l2_error(D, test), false);

  /*
  print("HQRRP");
  timing::start("HQRRP");
  Dense Dt(D);
  Dt.transpose();
  std::vector<int> jpvt(D.dim[0], 0);
  std::vector<double> tau(D.dim[0]);
  timing::start("DGEQP4");
  NoFLA_HQRRP_WY_blk_var4(D.dim[0], D.dim[1], &Dt, D.dim[0], jpvt.data(), tau.data(), 64, 5, 1);
  timing::stop("DGEQP4");
  Dense R(rank, D.dim[1]);
  for(int64_t i=0; i<D.dim[1]; i++) {
    for(int64_t j=0; j<D.dim[0]; j++) {
      if(j>=i)
        R(j, i) = Dt(j, i);
    }
  }
  timing::start("DORGQR");
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, D.dim[0], D.dim[1], rank, &Dt, D.stride, &tau[0]);
  timing::stop("DORGQR");
  timing::stopAndPrint("HQRRP", 2);
  Dense Result(D);
  for (int64_t i=0; i<Result.dim[1]; ++i)
    for (int64_t j=0; j<Result.dim[0]; ++j)
      Result(j, i) = D(j, jpvt[i]-1);
  print("Rel. L2 Error", l2_error(Result, gemm(Dt,R,1,true, false)), false);
  */
  /*
  Dense sing = get_singular_values(D);
  double err = 0.0;
  for (int i=rank; i<N; ++i)
    err += sing[i] * sing[i];
  print("Minimal possible error", std::sqrt(err), false);*/}
  return 0;
}
