#include "hicma/classes/initialization_helpers/matrix_initializer_file.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/cluster_tree.h"
#include "hicma/classes/initialization_helpers/index_range.h"

#include <cstdint>
#include <utility>
#include <cassert>
#include <fstream>


namespace hicma
{

// Additional constructors
MatrixInitializerFile::MatrixInitializerFile(
  std::string filename, MatrixLayout ordering, double admis, int64_t rank,
  int admis_type, const std::vector<std::vector<double>>& coords
) : MatrixInitializer(admis, rank, admis_type, coords),
    filename(filename), ordering(ordering) {}

void MatrixInitializerFile::fill_dense_representation(
  Dense& A, const IndexRange& row_range, const IndexRange& col_range
) const {
  std::ifstream file;
  file.open(filename);
  int64_t dim[2];
  double a;
  file >>dim[0] >>dim[1];
  assert(row_range.start+A.dim[0] <= dim[0]);
  assert(col_range.start+A.dim[1] <= dim[1]);
  if(ordering == HICMA_ROW_MAJOR) {
    for(int64_t i=0; i<std::min(dim[0], row_range.start+A.dim[0]); i++) {
      for(int64_t j=0; j<dim[1]; j++) {
        file >>a;
        if((i >= row_range.start && j >= col_range.start)
           && (i < row_range.start+A.dim[0] && j < col_range.start+A.dim[1])) {
          A(i-row_range.start, j-col_range.start) = a;
        }
      }
    }
  }
  else { //Assume entries in file are stored in column major
    for(int64_t j=0; j<std::min(dim[1], col_range.start+A.dim[1]); j++) {
      for(int64_t i=0; i<dim[0]; i++) {
        file >>a;
        if((i >= row_range.start && j >= col_range.start)
           && (i < row_range.start+A.dim[0] && j < col_range.start+A.dim[1])) {
          A(i-row_range.start, j-col_range.start) = a;
        }
      }
    }
  }
  file.close();
}

} // namespace hicma
