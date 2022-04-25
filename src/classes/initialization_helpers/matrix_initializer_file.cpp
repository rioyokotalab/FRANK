#include "FRANK/classes/initialization_helpers/matrix_initializer_file.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/initialization_helpers/cluster_tree.h"
#include "FRANK/classes/initialization_helpers/index_range.h"

#include <cstdint>
#include <utility>
#include <cassert>
#include <iostream>
#include <fstream>
#include <limits>


namespace FRANK
{

// Additional constructors
MatrixInitializerFile::MatrixInitializerFile(
  const std::string filename, const MatrixLayout ordering
) : MatrixInitializer(0, 0, 0, {}, AdmisType::PositionBased),
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
  switch(ordering) {
    case MatrixLayout::RowMajor:
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
      break;
    case MatrixLayout::ColumnMajor:
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
      break;
  }
  file.close();
}

} // namespace FRANK
