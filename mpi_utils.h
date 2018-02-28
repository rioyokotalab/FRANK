#ifndef mpi_utils_h
#define mpi_utils_h
#include <mpi.h>

namespace hicma {
  int MPIRANK;
  int MPISIZE;
  int BLCRANK;
  int BLCSIZE;
  int BLCROW;
  int BLCCOL;
  int EXTERNAL;

  void startMPI(int argc, char ** argv) {
    MPI_Initialized(&EXTERNAL);
    if (!EXTERNAL) MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &MPIRANK);
    MPI_Comm_size(MPI_COMM_WORLD, &MPISIZE);
  }

  void stopMPI() {
    if (!EXTERNAL) MPI_Finalize();
  }
}
#endif
