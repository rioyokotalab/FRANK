#include "hicma/classes/initialization_helpers/matrix_initializer_file.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/cluster_tree.h"
#include "hicma/classes/initialization_helpers/index_range.h"

#include <cstdint>
#include <utility>
#include <cassert>
#include <iostream>
#include <fstream>
#include <limits>


namespace hicma
{

// Additional constructors
MatrixInitializerFile::MatrixInitializerFile(
  std::string filename, MatrixLayout ordering, double admis, int64_t rank,
  std::vector<std::vector<double>> params, int admis_type
) : MatrixInitializer(admis, rank, params, admis_type),
    filename(filename), ordering(ordering) {}

void MatrixInitializerFile::fill_dense_representation(
  Dense& A, const IndexRange& row_range, const IndexRange& col_range
) const {
  std::ifstream file;
  file.open(filename);
  if(!file) {
    std::cout <<"Could not open matrix file: " <<filename <<" does not exist\n";
    return;
  }
  int64_t dim[2];
  double a;
  file >>dim[0] >>dim[1];
  file.ignore(1, '\n'); //Ignore newline after dim
  assert(row_range.start+A.dim[0] <= dim[0]);
  assert(col_range.start+A.dim[1] <= dim[1]);
  if(ordering == HICMA_ROW_MAJOR) {
    //Skip rows before row_start
    for(int64_t i=0; i<row_range.start; i++) {
      file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    for(int64_t i=0; i<A.dim[0]; i++) {
      //Skip columns before col_start
      for(int64_t j=0; j<col_range.start; j++) {
	file.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
      }
      for(int64_t j=0; j<A.dim[1]; j++) {
        file >>a;
	A(i, j) = a;
      }
      if(col_range.start+A.dim[1] < dim[1])
	file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //Ignore rest of the columns (if any)
    }
  }
  else { //Assume entries in file are stored in column major
    //Skip cols before col_start
    for(int64_t j=0; j<col_range.start; j++) {
      file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    for(int64_t j=0; j<A.dim[1]; j++) {
      //Skip rows before row_start
      for(int64_t i=0; i<row_range.start; i++) {
	file.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
      }
      for(int64_t i=0; i<A.dim[0]; i++) {
        file >>a;
	A(i, j) = a;
      }
      if(row_range.start+A.dim[0] < dim[0])
	file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //Ignore rest of the rows (if any)
    }
  }
  file.close();
}

} // namespace hicma
