#include "hicma/functions.h"

#include "hicma/classes/dense.h"
#include "hicma/operations/misc/misc.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <iostream>


namespace hicma
{

void zeros(
  Dense& A, [[maybe_unused]] std::vector<double>& x,
  [[maybe_unused]] int64_t i_begin, [[maybe_unused]] int64_t j_begin
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) = 0;
    }
  }
}

void identity(
  Dense& A, [[maybe_unused]] std::vector<double>& x,
  int64_t i_begin, int64_t j_begin
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) = i_begin+i == j_begin+j ? 1 : 0;
    }
  }
}

void random_normal(
  Dense& A, [[maybe_unused]] std::vector<double>& x,
  [[maybe_unused]] int64_t i_begin, [[maybe_unused]] int64_t j_begin
) {
  std::random_device rd;
  std::mt19937 gen(rd());
  // TODO Remove random seed when experiments end
  gen.seed(0);
  std::normal_distribution<double> dist(0.0, 1.0);
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) = dist(gen);
    }
  }
}

void random_uniform(
  Dense& A, [[maybe_unused]] std::vector<double>& x,
  [[maybe_unused]] int64_t i_begin, [[maybe_unused]] int64_t j_begin
) {
  std::random_device rd;
  std::mt19937 gen(rd());
  // TODO Remove random seed when experiments end
  gen.seed(0);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) = dist(gen);
    }
  }
}

void arange(
  Dense& A, [[maybe_unused]] std::vector<double>& x,
  [[maybe_unused]] int64_t i_begin, [[maybe_unused]] int64_t j_begin
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) = (double)(i*A.dim[1]+j);
    }
  }
}

void laplace1d(
  Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) = 1 / (
        std::abs(x[i+i_begin] - x[j+j_begin]) + 1e-3);
    }
  }
}

void cauchy2d(
  std::vector<double>& data,
  std::vector<std::vector<double>>& x,
  int64_t ni, int64_t nj,
  int64_t i_begin, int64_t j_begin
) {
  for (int64_t i=0; i<ni; i++) {
    for (int64_t j=0; j<nj; j++) {
      // double sgn = (arc4random() % 2 ? 1.0 : -1.0);
      double rij = (x[0][i+i_begin] - x[1][j+j_begin]) + 1e-2;
      data[i*nj+j] = 1.0 / rij;
    }
  }
}

void laplacend(
  std::vector<double>& data,
  std::vector<std::vector<double>>& x,
  int64_t ni, int64_t nj,
  int64_t i_begin, int64_t j_begin
) {
  for (int64_t i=0; i<ni; i++) {
    for (int64_t j=0; j<nj; j++) {
      double rij = 0.0;
      for(size_t k=0; k<x.size(); k++) {
        rij += (x[k][i+i_begin]-x[k][j+j_begin])*(x[k][i+i_begin]-x[k][j+j_begin]);
      }
      data[i*nj+j] = 1 / (std::sqrt(rij) + 1e-3);
    }
  }
}

void helmholtznd(
  std::vector< double>& data,
  std::vector<std::vector<double>>& x,
  int64_t ni, int64_t nj,
  int64_t i_begin, int64_t j_begin
) {
  for (int64_t i=0; i<ni; i++) {
    for (int64_t j=0; j<nj; j++) {
      double rij = 0.0;
      for(size_t k=0; k<x.size(); k++) {
        rij += (x[k][i+i_begin]-x[k][j+j_begin])*(x[k][i+i_begin]-x[k][j+j_begin]);
      }
      data[i*nj+j] = std::exp(-1.0 * rij) / (std::sqrt(rij) + 1e-3);
    }
  }
}

bool is_admissible_nd(
  std::vector<std::vector<double>>& x,
  int64_t ni, int64_t nj,
  int64_t i_begin, int64_t j_begin,
  double admis
) {
  std::vector<double> diamsI, diamsJ, centerI, centerJ;
  for(size_t k=0; k<x.size(); k++) {
    diamsI.push_back(diam(x[k], ni, i_begin));
    diamsJ.push_back(diam(x[k], nj, j_begin));
    centerI.push_back(mean(x[k], ni, i_begin));
    centerJ.push_back(mean(x[k], nj, j_begin));
  }
  double diamI = *std::max_element(diamsI.begin(), diamsI.end());
  double diamJ = *std::max_element(diamsJ.begin(), diamsJ.end());
  double dist = 0.0;
  for(size_t k=0; k<x.size(); k++) {
    dist += (centerI[k]-centerJ[k])*(centerI[k]-centerJ[k]);
  }
  dist = std::sqrt(dist);
  return (std::max(diamI, diamJ) <= (admis * dist));
}

bool is_admissible_nd_morton(
  std::vector<std::vector<double>>& x,
  int64_t ni, int64_t nj,
  int64_t i_begin, int64_t j_begin,
  double admis
) {
  std::vector<double> diamsI, diamsJ, centerI, centerJ;
  for(size_t k=0; k<x.size(); k++) {
    diamsI.push_back(diam(x[k], ni, i_begin));
    diamsJ.push_back(diam(x[k], nj, j_begin));
    centerI.push_back(mean(x[k], ni, i_begin));
    centerJ.push_back(mean(x[k], nj, j_begin));
  }
  double diamI = *std::max_element(diamsI.begin(), diamsI.end());
  double diamJ = *std::max_element(diamsJ.begin(), diamsJ.end());
  //Compute distance based on morton index of box
  int64_t boxSize = std::min(ni, nj);
  int64_t npartitions = x[0].size()/boxSize;
  int64_t level = (int64_t)log2((double)npartitions);
  std::vector<int64_t> indexI(x.size(), 0), indexJ(x.size(), 0);
  for(size_t k=0; k<x.size(); k++) {
    indexI[k] = i_begin/boxSize;
    indexJ[k] = j_begin/boxSize;
  }
  double dist = std::abs((double)getMortonIndex(indexI, level) - (double)getMortonIndex(indexJ, level));
  return (std::max(diamI, diamJ) <= (admis * dist));
}


} // namespace hicma
