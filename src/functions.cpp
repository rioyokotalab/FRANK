#include "hicma/functions.h"
#include "hicma/operations/misc.h"

#include <cmath>
#include <random>
#include <vector>
#include <iostream>

namespace hicma {

void zeros(
  std::vector<double>& data,
  std::vector<double>& x,
  const int& ni,
  const int& nj,
  const int& i_begin,
  const int& j_begin
) {
  for (int i=0; i<ni; i++) {
    for (int j=0; j<nj; j++) {
      data[i*nj+j] = 0;
    }
  }
}

void identity(
  std::vector<double>& data,
  std::vector<double>& x,
  const int& ni,
  const int& nj,
  const int& i_begin,
  const int& j_begin
) {
  for (int i=0; i<ni; i++) {
    for (int j=0; j<nj; j++) {
      data[i*nj+j] = i_begin+i == j_begin+j ? 1 : 0;
    }
  }
}

void random_normal(
  std::vector<double>& data,
  std::vector<double>& x,
  const int& ni,
  const int& nj,
  const int& i_begin,
  const int& j_begin
) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::normal_distribution<double> dist(0.0, 1.0);
  for (int i=0; i<ni; i++) {
    for (int j=0; j<nj; j++) {
      data[i*nj+j] = dist(mt);
    }
  }
}

void random_uniform(
  std::vector<double>& data,
  std::vector<double>& x,
  const int& ni,
  const int& nj,
  const int& i_begin,
  const int& j_begin
) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (int i=0; i<ni; i++) {
    for (int j=0; j<nj; j++) {
      data[i*nj+j] = dist(mt);
    }
  }
}

void arange(
  std::vector<double>& data,
  std::vector<double>& x,
  const int& ni,
  const int& nj,
  const int& i_begin,
  const int& j_begin
) {
  for (int i=0; i<ni; i++) {
    for (int j=0; j<nj; j++) {
      data[i*nj+j] = (double)(i*nj+j);
    }
  }
}

void laplace1d(
  std::vector<double>& data,
  std::vector<double>& x,
  const int& ni,
  const int& nj,
  const int& i_begin,
  const int& j_begin
) {
  for (int i=0; i<ni; i++) {
    for (int j=0; j<nj; j++) {
      data[i*nj+j] = 1 / (std::abs(x[i+i_begin] - x[j+j_begin]) + 1e-3);
    }
  }
}

void cauchy2d(
  std::vector<double>& data,
  std::vector<std::vector<double>>& x,
  const int& ni,
  const int& nj,
  const int& i_begin,
  const int& j_begin
) {
  for (int i=0; i<ni; i++) {
    for (int j=0; j<nj; j++) {
      // double sgn = (arc4random() % 2 ? 1.0 : -1.0);
      double rij = (x[0][i+i_begin] - x[1][j+j_begin]) + 1e-2;
      data[i*nj+j] = 1.0 / rij;
    }
  }
}

void laplacend(
  std::vector<double>& data,
  std::vector<std::vector<double>>& x,
  const int& ni,
  const int& nj,
  const int& i_begin,
  const int& j_begin
) {
  for (int i=0; i<ni; i++) {
    for (int j=0; j<nj; j++) {
      double rij = 0.0;
      for(int k=0; k<(int)x.size(); k++) {
        rij += (x[k][i+i_begin]-x[k][j+j_begin])*(x[k][i+i_begin]-x[k][j+j_begin]);
      }
      data[i*nj+j] = 1 / (std::sqrt(rij) + 1e-3);
    }
  }
}

void helmholtznd(
  std::vector< double>& data,
  std::vector<std::vector<double>>& x,
  const int& ni,
  const int& nj,
  const int& i_begin,
  const int& j_begin
) {
  for (int i=0; i<ni; i++) {
    for (int j=0; j<nj; j++) {
      double rij = 0.0;
      for(int k=0; k<(int)x.size(); k++) {
        rij += (x[k][i+i_begin]-x[k][j+j_begin])*(x[k][i+i_begin]-x[k][j+j_begin]);
      }
      data[i*nj+j] = std::exp(-1.0 * rij) / (std::sqrt(rij) + 1e-3);
    }
  }
}

bool is_admissible_nd(
  std::vector<std::vector<double>>& x,
  const int& ni,
  const int& nj,
  const int& i_begin,
  const int& j_begin,
  const double& admis
) {
  std::vector<double> diamsI, diamsJ, centerI, centerJ;
  for(int k=0; k<(int)x.size(); k++) {
    diamsI.push_back(diam(x[k], ni, i_begin));
    diamsJ.push_back(diam(x[k], nj, j_begin));
    centerI.push_back(mean(x[k], ni, i_begin));
    centerJ.push_back(mean(x[k], nj, j_begin));
  }
  double diamI = *std::max_element(diamsI.begin(), diamsI.end());
  double diamJ = *std::max_element(diamsJ.begin(), diamsJ.end());
  double dist = 0.0;
  for(int k=0; k<(int)x.size(); k++) {
    dist += (centerI[k]-centerJ[k])*(centerI[k]-centerJ[k]);
  }
  dist = std::sqrt(dist);
  return (std::max(diamI, diamJ) <= (admis * dist));
}

bool is_admissible_nd_morton(
  std::vector<std::vector<double>>& x,
  const int& ni,
  const int& nj,
  const int& i_begin,
  const int& j_begin,
  const double& admis
) {
  std::vector<double> diamsI, diamsJ, centerI, centerJ;
  for(int k=0; k<(int)x.size(); k++) {
    diamsI.push_back(diam(x[k], ni, i_begin));
    diamsJ.push_back(diam(x[k], nj, j_begin));
    centerI.push_back(mean(x[k], ni, i_begin));
    centerJ.push_back(mean(x[k], nj, j_begin));
  }
  double diamI = *std::max_element(diamsI.begin(), diamsI.end());
  double diamJ = *std::max_element(diamsJ.begin(), diamsJ.end());
  //Compute distance based on morton index of box
  int boxSize = std::min(ni, nj);
  int npartitions = x[0].size()/boxSize;
  int level = (int)log2((double)npartitions);
  std::vector<int> indexI(x.size(), 0), indexJ(x.size(), 0);
  for(int k=0; k<(int)x.size(); k++) {
    indexI[k] = i_begin/boxSize;
    indexJ[k] = j_begin/boxSize;
  }
  double dist = std::abs((double)getMortonIndex(indexI, level) - (double)getMortonIndex(indexJ, level));
  return (std::max(diamI, diamJ) <= (admis * dist));
}


} // namespace hicma
