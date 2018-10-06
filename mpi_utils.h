#ifndef mpi_utils_h
#define mpi_utils_h
#include <mpi.h>

namespace hicma {
  extern int MPIRANK;
  extern int MPISIZE;
  extern int CONTEXT;
  extern int ROWSIZE;
  extern int COLSIZE;
  extern int ROWRANK;
  extern int COLRANK;
  extern int EXTERNAL;

  void startMPI(int argc, char ** argv);

  void stopMPI();
}
#endif
