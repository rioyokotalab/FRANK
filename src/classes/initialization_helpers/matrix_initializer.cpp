#include "hicma/classes/initialization_helpers/matrix_initializer.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/initialization_helpers/cluster_tree.h"
#include "hicma/functions.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/misc.h"
#include "hicma/operations/randomized_factorizations.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>
#include <cmath>


namespace hicma
{

MatrixInitializer::MatrixInitializer(
  double admis, double eps, int64_t rank,
  std::vector<std::vector<double>> params, AdmisType admis_type
) : admis(admis), eps(eps), rank(rank),
    params(params), admis_type(admis_type) {}

Dense MatrixInitializer::get_dense_representation(
  const ClusterTree& node
) const {
  Dense representation(node.rows.n, node.cols.n);
  fill_dense_representation(representation, node.rows, node.cols);
  return representation;
}

LowRank MatrixInitializer::get_compressed_representation(
  const ClusterTree& node, bool fixed_rank
) {
  // TODO This function still relies on ClusterTree to be symmetric!
  if(fixed_rank) return LowRank(get_dense_representation(node), rank);
  else return LowRank(get_dense_representation(node), eps);
}

std::vector<std::vector<double>> MatrixInitializer::get_coords_range(const IndexRange& range) const {
  std::vector<std::vector<double>> coords_range;
  for(size_t d=0; d<params.size(); d++)
    coords_range.push_back(std::vector<double>(params[d].begin()+range.start, params[d].begin()+range.start+range.n));
  return coords_range;
}

bool MatrixInitializer::is_admissible(const ClusterTree& node) const {
  bool admissible = true;
  // Vectors are never admissible
  admissible &= (node.rows.n > 1 && node.cols.n > 1);
  if(admis_type == PositionBasedAdmis)
    admissible &= (node.dist_to_diag() > (int64_t)admis);
  else {
    //Get actual coordinates
    std::vector<std::vector<double>> row_coords = get_coords_range(node.rows);
    std::vector<std::vector<double>> col_coords = get_coords_range(node.cols);
    //Calculate bounding boxes
    double offset = 5e-1;
    std::vector<double> max_coord_row, min_coord_row, center_coord_row;
    std::vector<double> max_coord_col, min_coord_col, center_coord_col;
    for(size_t d=0; d<row_coords.size(); d++) {
      min_coord_row.push_back(-offset + *std::min_element(row_coords[d].begin(), row_coords[d].end()));
      max_coord_row.push_back(offset + *std::max_element(row_coords[d].begin(), row_coords[d].end()));
      center_coord_row.push_back(min_coord_row[d] + (max_coord_row[d]-min_coord_row[d])/2.0);

      min_coord_col.push_back(-offset + *std::min_element(col_coords[d].begin(), col_coords[d].end()));
      max_coord_col.push_back(offset + *std::max_element(col_coords[d].begin(), col_coords[d].end()));
      center_coord_col.push_back(min_coord_col[d] + (max_coord_col[d]-min_coord_col[d])/2.0);
    }
    //Calculate diameter and distance
    double max_length_row = 0.0;
    double max_length_col = 0.0;
    double dist = 0.0;
    for(size_t k=0; k<row_coords.size(); k++) {
      max_length_row = std::max(max_length_row, max_coord_row[k] - min_coord_row[k]);
      max_length_col = std::max(max_length_col, max_coord_col[k] - min_coord_col[k]);
      double d = std::fabs(center_coord_row[k] - center_coord_col[k]);
      dist += d * d;
    }
    double diam = std::max(max_length_row, max_length_col);
    admissible &= ((admis * admis * diam * diam) < dist);
  }
  return admissible;
}

} // namespace hicma
