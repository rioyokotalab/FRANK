#include "hicma/classes/initialization_helpers/matrix_initializer_kernel.h"

#include "hicma/classes/initialization_helpers/index_range.h"
#include "hicma/classes/initialization_helpers/cluster_tree.h"

#include <algorithm>
#include <cmath>
#include <utility>


namespace hicma
{

// explicit template initialization (these are the only available types)
template class MatrixInitializerKernel<float>;
template class MatrixInitializerKernel<double>;

template<typename U>
MatrixInitializerKernel<U>::MatrixInitializerKernel(
    const MatrixKernel<U>& kernel, double admis, int64_t rank, int admis_type
) : MatrixInitializer(admis, rank, admis_type), kernel_ptr(kernel.clone()) {}

template<typename U>
MatrixInitializerKernel<U>::MatrixInitializerKernel(
    MatrixKernel<U>&& kernel, double admis, int64_t rank, int admis_type
) : MatrixInitializer(admis, rank, admis_type), kernel_ptr(kernel.move_clone()) {}

template<typename U>
void MatrixInitializerKernel<U>::fill_dense_representation(
  Matrix& A, const IndexRange& row_range, const IndexRange& col_range
) const {
  (*kernel_ptr).apply(A, row_range.start, col_range.start);
}

template<typename U>
bool MatrixInitializerKernel<U>::is_admissible(const ClusterTree& node) const {
  bool admissible = true;
  // Vectors are never admissible
  admissible &= (node.rows.n > 1 && node.cols.n > 1);
  if(admis_type == POSITION_BASED_ADMIS) {
    admissible &= (node.dist_to_diag() > (int64_t)admis);
  }
  else {
    //Get actual coordinates
    std::vector<std::vector<U>> row_coords = (*kernel_ptr).get_coords_range(node.rows);
    std::vector<std::vector<U>> col_coords = (*kernel_ptr).get_coords_range(node.cols);
    //Calculate bounding boxes
    U offset = 5e-1;
    std::vector<U> max_coord_row, min_coord_row, center_coord_row;
    std::vector<U> max_coord_col, min_coord_col, center_coord_col;
    for(size_t d=0; d<row_coords.size(); d++) {
      min_coord_row.push_back(-offset + *std::min_element(row_coords[d].begin(), row_coords[d].end()));
      max_coord_row.push_back(offset + *std::max_element(row_coords[d].begin(), row_coords[d].end()));
      center_coord_row.push_back(min_coord_row[d] + (max_coord_row[d]-min_coord_row[d])/2.0);

      min_coord_col.push_back(-offset + *std::min_element(col_coords[d].begin(), col_coords[d].end()));
      max_coord_col.push_back(offset + *std::max_element(col_coords[d].begin(), col_coords[d].end()));
      center_coord_col.push_back(min_coord_col[d] + (max_coord_col[d]-min_coord_col[d])/2.0);
    }
    //Calculate diameter and distance
    U max_length_row = 0.0;
    U max_length_col = 0.0;
    U dist = 0.0;
    for(size_t k=0; k<row_coords.size(); k++) {
      max_length_row = std::max(max_length_row, max_coord_row[k] - min_coord_row[k]);
      max_length_col = std::max(max_length_col, max_coord_col[k] - min_coord_col[k]);
      U d = std::fabs(center_coord_row[k] - center_coord_col[k]);
      dist += d * d;
    }
    U diam = std::max(max_length_row, max_length_col);
    admissible &= ((admis * admis * diam * diam) < dist);
  }
  return admissible;
}

} // namespace hicma
