#include "catch.hpp"
#include "mpi_utils.h"
#include "functions.h"
#include "timer.h"
#include "node.h"
#include "hierarchical.h"

using namespace hicma;

TEST_CASE("Simple dense matrix block splitting and creation", "dense-only") {
  startMPI(0, NULL);

  int N = 64;
  int NB = 32;
  int P = 2;
  int Q = 2;

  hicma::ROWRANK = (int)(MPIRANK / P);
  hicma::COLRANK = MPIRANK % Q;

  SECTION("single matrix consumes all memory") {
    Hierarchical A(N, N, NB, NB, P, Q, MPI_COMM_WORLD);
    std::vector<double> randx(N*N);
    for (int i = 0; i < N*N; ++i) {
      randx[i] = drand48();
    }

    REQUIRE ( A.get_map().empty() );

    // this will basically duplicate the entire N*N matrix across P*Q processes.
    // the create_dense_block in this case will populate the entire Hierarchical
    //   structure with one huge dense block.
    A.create_dense_block(randx);

    // TODO: comparisons can only happen between blocks. Not between raw data and
    // matrix types.
    REQUIRE ( A.get_data() == randx );
  }

  SECTION("split blocks on single process") {
    Hierarchical A(N, N, NB, NB, P, Q, MPI_COMM_WORLD);

    REQUIRE ( A.get_map().empty() );

    // supply process number on which all the blocks should lie.
    A.single_process_split(0);

    // verify that all entries in map point to process 0.
    // Blocks will need to be queried individually for obtaining an individual value.
    if (MPIRANK == 0) {
      REQUIRE ( A.has_block(0,0) == true );
      REQUIRE ( A.has_block(0,1) == true );
      REQUIRE ( A.has_block(1,0) == true );
      REQUIRE ( A.has_block(1,1) == true );
    }
    else {
      REQUIRE ( A.has_block(0,0) == false );
      REQUIRE ( A.has_block(0,1) == false );
      REQUIRE ( A.has_block(1,0) == false );
      REQUIRE ( A.has_block(1,1) == false );
    }

    std::vector<double> randx(NB*NB);
    for (int i = 0; i < NB*NB; ++i) {
      randx[i] = drand48();
    }
    
    // The () operator on Hierarchical will return the block at the particular position
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < 2; ++col) {
        if (A.has_block(row,col)) {
          // the create_dense_block function will allocate memory and set data for a dense block.
          A(row,col).create_dense_block(randx);
        }        
      }
    }

    // since all the blocks are on the same process (0) they should contain
    // the same random data.
    if (MPIRANK == 0) {
      REQUIRE ( A(0,0).get_data() == A(0,1).get_data() );
      REQUIRE ( A(0,1).get_data() == A(1,0).get_data() );
      REQUIRE ( A(1,1).get_data() == A(0,0).get_data() );
      REQUIRE ( A(1,0).get_data() == A(1,1).get_data() );
    }

    std::vector<double> big_randx(N*N);
    // trying to populate a split matrix should throw an execetion
    CHECK_THROWS_AS(A.create_dense_block(big_randx), DenseBlockCreationError);
  }
  
  SECTION("1-level split blocks accross multiple processes") {
    Hierarchical A(N, N, NB, NB, P, Q, MPI_COMM_WORLD);

    // initialize map to nil so this matrix is not split yet.
    REQUIRE ( A.get_map().empty() );

    // split the matrix across P*Q processes.
    A.multi_process_split();

    // Now matrix looks like so:
    // |--------|---------|
    // |   0    |   1     |
    // |        |         |
    // |--------|---------|
    // |   2    |   3     |
    // |        |         |
    // |--------|---------|
    
    std::vector<double> randx(NB*NB);
    for (int i = 0; i < NB*NB; ++i) {
      randx[i] = drand48();
    }

    // each process should have a block in the same indices corresponding to its rank
    //  since we have split the block evenly across all processes.
    REQUIRE (A.has_block(ROWRANK, COLRANK) == true);

    // Populate each block with a random vector. should be distinct for each process.
    A(ROWRANK, COLRANK).create_dense_block(randx);

    // This process should not posses blocks belonging to some other process.
    REQUIRE (A.has_block((ROWRANK + 1) % P, (COLRANK + 1) % Q) == false);
  }

  SECTION("2-level split blocks across multiple processes") {
    std::vector<double> randx((NB/2) * (NB/2));
    for (int i = 0; i < (NB/2) * (NB/2); ++i) {
      randx[i] = drand48();
    }

    Hierarchical A(N, N, NB, NB, P, Q, MPI_COMM_WORLD);

    // 1-level split across multiple processes.
    A.multi_process_split();
    // 2-level split across multiple processes.
    A(ROWRANK, COLRANK).multi_process_split();

    // now matrix structure looks like so:
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

    // trying to populate a multi-level split matrix should throw an exception
    CHECK_THROWS_AS(A.create_dense_block(), DenseBlockCreationError);

    // populate each level 2 block
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        if (A(ROWRANK, COLRANK).has_block(i, j)) {
          A(ROWRANK, COLRANK)(i,j).create_dense_block(randx);
        }
      }
    }

    // verify that blocks meant for one process don't exist in another
    if (MPIRANK == 0) {
      REQUIRE( A.has_block(0,0) == true );
      REQUIRE( A(0,0).has_block(0,0) == true );
      REQUIRE( A(0,1).has_block(0,0) == true );
      REQUIRE( A(1,0).has_block(0,0) == true );
      REQUIRE( A(1,1).has_block(0,0) == true  );

      REQUIRE( A.has_block(0,1) == false );
      REQUIRE( A(0,0).has_block(0,1) == false );
      REQUIRE( A(0,1).has_block(0,1) == false );
      REQUIRE( A(1,0).has_block(0,1) == false );
      REQUIRE( A(1,1).has_block(0,1) == false );

      REQUIRE( A.has_block(1,0) == false );
      REQUIRE( A(0,0).has_block(1,0) == false );
      REQUIRE( A(0,1).has_block(1,0) == false );
      REQUIRE( A(1,0).has_block(1,0) == false );
      REQUIRE( A(1,1).has_block(1,0) == false );

      REQUIRE( A.has_block(1,1) == false );
      REQUIRE( A(0,0).has_block(1,1) == false );
      REQUIRE( A(0,1).has_block(1,1) == false );
      REQUIRE( A(1,0).has_block(1,1) == false );
      REQUIRE( A(1,1).has_block(1,1) == false );
    }

    if (MPIRANK == 3) {
      REQUIRE( A.has_block(0,0) == false );
      REQUIRE( A(0,0).has_block(0,0) == false );
      REQUIRE( A(0,1).has_block(0,0) == false );
      REQUIRE( A(1,0).has_block(0,0) == false );
      REQUIRE( A(1,1).has_block(0,0) == false  );

      REQUIRE( A.has_block(0,1) == false );
      REQUIRE( A(0,0).has_block(0,1) == false );
      REQUIRE( A(0,1).has_block(0,1) == false );
      REQUIRE( A(1,0).has_block(0,1) == false );
      REQUIRE( A(1,1).has_block(0,1) == false );

      REQUIRE( A.has_block(1,0) == false );
      REQUIRE( A(0,0).has_block(1,0) == false );
      REQUIRE( A(0,1).has_block(1,0) == false );
      REQUIRE( A(1,0).has_block(1,0) == false );
      REQUIRE( A(1,1).has_block(1,0) == false );

      REQUIRE( A.has_block(1,1) == true );
      REQUIRE( A(0,0).has_block(1,1) == true );
      REQUIRE( A(0,1).has_block(1,1) == true );
      REQUIRE( A(1,0).has_block(1,1) == true );
      REQUIRE( A(1,1).has_block(1,1) == true );      
    }
  }

  SECTION("2-level split allocated blocks across processes") {
    Hierarchical A(N, N, NB, NB, P, Q, MPI_COMM_WORLD);
    std::vector<double> randx(NB*NB);
    for (int i = 0; i < NB*NB; ++i) {
      randx[i] = drand48();
    }
    
    A.multi_process_split();
    A(ROWRANK, COLRANK).create_dense_block(randx);

    // This function should distribute the data already present in one process
    //   block across all the other processes.
    A(ROWRANK, COLRANK).multi_process_split();

    // since we distribute the same random array across all process the parts of
    //   the array possessed by each process after the 2-level split should be the same.
    if (MPIRANK == 0) {
      REQUIRE( A(0,0)(0,0).get_data() == A(1,1)(0,0).get_data() );
    }

    if (MPIRANK == 3) {
      REQUIRE( A(1,1)(1,1).get_data() == A(1,0)(1,1).get_data() );
    }
  }

  stopMPI();
}
