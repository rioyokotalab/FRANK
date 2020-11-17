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

MatrixInitializer::MatrixInitializer(int64_t admis, int64_t rank)
: admis(admis), rank(rank) {}

LowRank MatrixInitializer::get_compressed_representation(
  const ClusterTree& node
) {
  // TODO This function still relies on ClusterTree to be symmetric!
  return LowRank(get_dense_representation(node), rank);
}

bool MatrixInitializer::is_admissible(
  const ClusterTree& node,
  const std::vector<std::vector<double>>& x,
  int admis_type
) const {
  bool admissible = true;
  if(admis_type == POSITION_BASED_ADMIS)
    admissible &= position_based_admissible(node);
  else
    admissible &= geometry_based_admissible(node, x);
  // Vectors are never admissible
  admissible &= (node.rows.n > 1 && node.cols.n > 1);
  return admissible;
}

bool MatrixInitializer::position_based_admissible(const ClusterTree& node) const {
  return (node.dist_to_diag() > admis);
}

bool MatrixInitializer::geometry_based_admissible(
  const ClusterTree& node,
  const std::vector<std::vector<double>>& x
) const {
  //Calculate bounding boxes
  double offset = 5e-1;
  std::vector<double> xmax_row, xmin_row, center_row;
  std::vector<double> xmax_col, xmin_col, center_col;
  for(size_t k=0; k<x.size(); k++) {
    xmin_row.push_back(-offset + *std::min_element(x[k].begin()+node.rows.start, x[k].begin()+node.rows.start+node.rows.n));
    xmax_row.push_back(offset + *std::max_element(x[k].begin()+node.rows.start, x[k].begin()+node.rows.start+node.rows.n));
    center_row.push_back(xmin_row[k] + (xmax_row[k]-xmin_row[k])/2.0);

    xmin_col.push_back(-offset + *std::min_element(x[k].begin()+node.cols.start, x[k].begin()+node.cols.start+node.cols.n));
    xmax_col.push_back(offset + *std::max_element(x[k].begin()+node.cols.start, x[k].begin()+node.cols.start+node.cols.n));
    center_col.push_back(xmin_col[k] + (xmax_col[k]-xmin_col[k])/2.0);
  }
  //Calculate diameter and distance
  double diam_row = 0.0;
  double diam_col = 0.0;
  double dist = 0.0;
  for(size_t k=0; k<x.size(); k++) {
    diam_row = std::max(diam_row, xmax_row[k] - xmin_row[k]);
    diam_col = std::max(diam_col, xmax_col[k] - xmin_col[k]);
    double d = std::fabs(center_row[k] - center_col[k]);
    dist += d * d;
  }
  double diams = std::max(diam_row, diam_col);
  return ((diams * diams) <= (admis * admis * dist));
}

} // namespace hicma
