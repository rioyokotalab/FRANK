#include "catch.hpp"
#include "mpi_utils.h"
#include "functions.h"
#include "timer.h"
#include "hierarchical.h"

using namespace hicma;

TEST_CASE("LU decomposition of tiled distributed dense matrix") {
  startMPI(0, NULL);

  int N = 64;
  int NB = 32;
  int P = 2;
  int Q = 2;

  // This test targets a matrix like so:
  // 
  // |-------------------|\
  // ||---|---|||---|---|| \      \        \
  // || 0 | 0 ||| 0 | 0 ||  \      \       | = NB / 2
  // ||---|---|||---|---||  |      | = NB  /
  // || 0 | 0 ||| 0 | 0 ||  |      /
  // ||---|---|||---|---||  | = N /
  // ||---|---|||---|---||  |
  // || 0 | 0 ||| 0 | 0 ||  |
  // ||---|---|||---|---||  /
  // || 0 | 0 ||| 0 | 0 || /
  // |-------------------|/
  SECTION("simple single process block LU") {
    Hierarchical A(N, N, NB, NB, P, Q, MPI_COMM_WORLD);

    A.single_process_split(0);

    if (MPIRANK == 0) {
      REQUIRE ( A.has_block(0,0) == true );

      std::vector<double> randx((NB/2) * (NB/2));


      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          A(i,j).single_process_split(0);
        }
      }

      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          for (int m = 0; m < 2; ++m) {
            for (int n = 0; m < 2; ++n) {
              
              for (int k = 0; k < (NB/2)*(NB/2); ++k) {
                randx[k] = drand48();
              }
              A(i,j)(m,n).create_dense_block(randx);
            }
          }
        }
      }
    }

    // the process that contains the matrix will participate in the LU decomposition.
    A.block_lu();
  }

  // This test targets a matrix like so:
  // 
  // |-------------------|\
  // ||---|---|||---|---|| \      \        \
  // || 0 | 1 ||| 0 | 1 ||  \      \       | = NB / 2
  // ||---|---|||---|---||  |      | = NB  /
  // || 2 | 3 ||| 2 | 3 ||  |      /
  // ||---|---|||---|---||  | = N /
  // ||---|---|||---|---||  |
  // || 0 | 1 ||| 0 | 1 ||  |
  // ||---|---|||---|---||  /
  // || 2 | 3 ||| 2 | 3 || /
  // |-------------------|/
  SECTION("multi process block LU") {
    Hierarchical A(N, N, NB, NB, P, Q, MPI_COMM_WORLD);
    std::vector<double> randx(NB*NB);
    for (int i = 0; i < NB*NB; ++i) {
      randx[i] = drand48();
    }

    A.multi_process_split();
    A(ROWRANK, COLRANK).create_dense_block(randx);
    A(ROWRANK, COLRANK).multi_process_split();

    A.block_lu();
  }

  stopMPI();
}
