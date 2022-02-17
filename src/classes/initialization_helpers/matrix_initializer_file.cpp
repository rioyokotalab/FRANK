#include "hicma/classes/initialization_helpers/matrix_initializer_file.h"

#include "hicma/classes/initialization_helpers/index_range.h"
#include "hicma/extension_headers/util.h"

#include <cassert>
#include <iostream>
#include <fstream>
#include <limits>
#include <typeinfo>


namespace hicma
{

// explicit template initialization (these are the only available types)
template class MatrixInitializerFile<float>;
template class MatrixInitializerFile<double>;

// Additional constructors
template<typename U>
MatrixInitializerFile<U>::MatrixInitializerFile(
  std::string filename, MatrixLayout ordering, double admis, int64_t rank)
  : MatrixInitializer(admis, rank),
    filename(filename), ordering(ordering) {}

template<typename U>
void MatrixInitializerFile<U>::fill_dense_representation(
  Matrix& A, const IndexRange& row_range, const IndexRange& col_range
) const {
  try {
    if (is_double(A)) {
      fill_dense_representation(dynamic_cast<Dense<double>&>(A), row_range, col_range);
    }
    else {
      fill_dense_representation(dynamic_cast<Dense<float>&>(A), row_range, col_range);
    }
  }
  catch(std::bad_cast& e) {
    // TODO better error handling
    std::cerr<<"MatrixInitializerFile: Could not initialize a non dense matrix."<<std::endl;
    std::abort();
  }
}

template<typename U> template<typename T>
void MatrixInitializerFile<U>::fill_dense_representation(
  Dense<T>& A, const IndexRange& row_range, const IndexRange& col_range
) const {
  std::ifstream file;
  file.open(filename);
  if(!file) {
    std::cout <<"Could not open matrix file: " <<filename <<" does not exist\n";
    return;
  }
  int64_t dim[2];
  U a;
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
        // relies on implicit type conversion
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
        // relies on implicit type conversion
	      A(i, j) = a;
      }
      if(row_range.start+A.dim[0] < dim[0])
	      file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //Ignore rest of the rows (if any)
    }
  }
  file.close();
}

} // namespace hicma
