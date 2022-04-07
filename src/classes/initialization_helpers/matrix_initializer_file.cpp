#include "hicma/classes/initialization_helpers/matrix_initializer_file.h"

#include "hicma/classes/matrix.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/index_range.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <iostream>
#include <fstream>
#include <limits>


namespace hicma
{

// Additional constructors
MatrixInitializerFile::MatrixInitializerFile(
  std::string filename, MatrixLayout ordering, double admis, int64_t rank, int admis_type, vec2d<double> params)
  : MatrixInitializer(admis, rank, admis_type, params),
    filename(filename), ordering(ordering) {}

declare_method(void, fill_from_file, (const std::string&, MatrixLayout, virtual_<Matrix&>, int64_t, int64_t))

void MatrixInitializerFile::fill_dense_representation(
  Matrix& A, const IndexRange& row_range, const IndexRange& col_range
) const {
  fill_from_file(filename, ordering, A, row_range.start, col_range.start);
}

template<typename T>
void fill_dense_from_file(
  const std::string& filename, MatrixLayout ordering,
  Dense<T>& A, int64_t row_start, int64_t col_start
) {
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
  assert(row_start+A.dim[0] <= dim[0]);
  assert(col_start+A.dim[1] <= dim[1]);
  if(ordering == HICMA_ROW_MAJOR) {
    //Skip rows before row_start
    for(int64_t i=0; i<row_start; i++) {
      file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    for(int64_t i=0; i<A.dim[0]; i++) {
      //Skip columns before col_start
      for(int64_t j=0; j<col_start; j++) {
	      file.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
      }
      for(int64_t j=0; j<A.dim[1]; j++) {
        file >>a;
        // relies on implicit type conversion
	      A(i, j) = a;
      }
      if(col_start+A.dim[1] < dim[1])
	      file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //Ignore rest of the columns (if any)
    }
  }
  else { //Assume entries in file are stored in column major
    //Skip cols before col_start
    for(int64_t j=0; j<col_start; j++) {
      file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    for(int64_t j=0; j<A.dim[1]; j++) {
      //Skip rows before row_start
      for(int64_t i=0; i<row_start; i++) {
	      file.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
      }
      for(int64_t i=0; i<A.dim[0]; i++) {
        file >>a;
        // relies on implicit type conversion
	      A(i, j) = a;
      }
      if(row_start+A.dim[0] < dim[0])
	      file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //Ignore rest of the rows (if any)
    }
  }
  file.close();
}

define_method(void, fill_from_file, (const std::string& filename, MatrixLayout ordering, Dense<float>& A, int64_t row_start, int64_t col_start)) {
  fill_dense_from_file(filename, ordering, A, row_start, col_start);
}

define_method(void, fill_from_file, (const std::string& filename, MatrixLayout ordering, Dense<double>& A, int64_t row_start, int64_t col_start)) {
  fill_dense_from_file(filename, ordering, A, row_start, col_start);
}

define_method(void, fill_from_file, (const std::string&, MatrixLayout, Matrix& A, int64_t, int64_t)) {
  omm_error_handler("fill_from_file", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
