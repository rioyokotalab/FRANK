#include "hicma/util/experiment_setup.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>


namespace hicma
{

template<typename U>
void sort_coords(std::vector<std::vector<U>>& coords, int dim, int start, int end);

// explicit template initialization (these are the only available types)
template std::vector<float> get_sorted_random_vector(int64_t N, int seed);
template std::vector<double> get_sorted_random_vector(int64_t N, int seed);
template std::vector<float> get_non_negative_vector(int64_t N);
template std::vector<double> get_non_negative_vector(int64_t N);
template std::vector<std::vector<float>> get_circular_coords(int64_t N, int);
template std::vector<std::vector<double>> get_circular_coords(int64_t N, int);
template std::vector<std::vector<float>> get_rectangular_coords(int64_t N, int);
template std::vector<std::vector<double>> get_rectangular_coords(int64_t N, int);
template std::vector<std::vector<float>> get_rectangular_coords_rand(int64_t N, int);
template std::vector<std::vector<double>> get_rectangular_coords_rand(int64_t N, int);
template void sort_coords(std::vector<std::vector<double>>& coords, int dim, int start, int end);
template void sort_coords(std::vector<std::vector<float>>& coords, int dim, int start, int end);


template<typename U>
std::vector<std::vector<U>> get_circular_coords(int64_t N, int ndim) {
  assert (ndim > 1 && ndim <4);
  const double PI = std::atan(1.0)*4;
  std::vector<std::vector<U>> result;

  if (ndim == 2) {
    std::vector<U> x_coords;
    std::vector<U> y_coords;
    // Generate a unit circle with N points on the circumference.
    for (int64_t i = 0; i < N; i++) {
        const double theta = (i * 2.0 * PI) / (double)N;
        x_coords.push_back(std::cos(theta));
        y_coords.push_back(std::sin(theta));
      }
    result.push_back(x_coords);
    result.push_back(y_coords);
  } else {
    assert(ndim == 3);
    std::vector<U> x_coords;
    std::vector<U> y_coords;
    std::vector<U> z_coords;
    // Generate a unit sphere mesh with N uniformly spaced points on the surface
    // https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    const double phi = PI * (3. - std::sqrt(5.));  // golden angle in radians
    for (int64_t i = 0; i < N; i++) {
      const double y = 1. - ((double)i / ((double)N - 1)) * 2.;  // y goes from 1 to -1

      // Note: setting constant radius = 1 will produce a cylindrical shape
      const double radius = std::sqrt(1. - y * y);  // radius at y
      const double theta = (double)i * phi;

      x_coords.push_back(radius * std::cos(theta));
      y_coords.push_back(y);
      z_coords.push_back(radius * std::sin(theta));
    }
    result.push_back(x_coords);
    result.push_back(y_coords);
    result.push_back(z_coords);
  }

  sort_coords(result, 0, 0, N);

  return result;
}

template<typename U>
std::vector<std::vector<U>> get_rectangular_coords_rand(int64_t N, int ndim) {
  assert (ndim > 1 && ndim <4);
  std::vector<std::vector<U>> result;
  std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(0);
  std::uniform_real_distribution<> dis(-1, 1);

  if (ndim > 1) {
    std::vector<U> x_coords;
    std::vector<U> y_coords;

    for (int64_t i=0; i<N; ++i) {
      x_coords.push_back(dis(gen));
      y_coords.push_back(dis(gen));
    }
    result.push_back(x_coords);
    result.push_back(y_coords);
  } 
  if (ndim > 2) {
    std::vector<U> z_coords;

    for (int64_t i=0; i<N; ++i) {
      z_coords.push_back(dis(gen));
    }
    result.push_back(z_coords);
  }

  sort_coords(result, 0, 0, N);

  return result;
}

template<typename U>
std::vector<std::vector<U>> get_rectangular_coords(int64_t N, int ndim) {
  assert (ndim > 1 && ndim <4);
  assert (N >= 4);
  std::vector<std::vector<U>> result;

  if (ndim == 2) {
    std::vector<U> x_coords;
    std::vector<U> y_coords;

    // Taken from H2Lib: Library/curve2d.c
    const double a = 1;
    const int64_t top = N / 4;
    const int64_t left = N / 2;
    const int64_t bottom = 3 * N / 4;
    int64_t i = 0;
    for (i = 0; i < top; i++) {
      x_coords.push_back(a - 2.0 * a * i / top);
      y_coords.push_back(a);
    }
    for (; i < left; i++) {
      x_coords.push_back(-a);
      y_coords.push_back(a - 2.0 * a * (i - top) / (left - top));
    }
    for (; i < bottom; i++) {
      x_coords.push_back(-a + 2.0 * a * (i - left) / (bottom - left));
      y_coords.push_back(-a);
    }
    for (; i < N; i++) {
      x_coords.push_back(a);
      y_coords.push_back(-a + 2.0 * a * (i - bottom) / (N - bottom));
    }

    result.push_back(x_coords);
    result.push_back(y_coords);
  } else {
    // Took this from Hatrix, but I am not sure if it works correctly
    assert(false);
    std::vector<U> x_coords;
    std::vector<U> y_coords;
    std::vector<U> z_coords;
    // Generate a unit cube mesh with N points around the surface
    const int64_t mlen = (int64_t)ceil((double)N / 6.);
    const double alen = std::sqrt((double)mlen);
    const int64_t m = (int64_t)std::ceil(alen);
    const int64_t n = (int64_t)std::ceil((double)mlen / m);

    const double seg_fv = 1. / ((double)m - 1);
    const double seg_fu = 1. / (double)n;
    const double seg_sv = 1. / ((double)m + 1);
    const double seg_su = 1. / ((double)n + 1);

    for (int64_t i = 0; i < N; i++) {
      const int64_t face = i / mlen;
      const int64_t ii = i - face * mlen;
      const int64_t x = ii / m;
      const int64_t y = ii - x * m;
      const int64_t x2 = y & 1;

      double u, v;
      double px, py, pz;

      switch (face) {
        case 0: // POSITIVE X
          v = y * seg_fv;
          u = (0.5 * x2 + x) * seg_fu;
          px = 1.;
          py = 2. * v - 1.;
          pz = -2. * u + 1.;
          break;
        case 1: // NEGATIVE X
          v = y * seg_fv;
          u = (0.5 * x2 + x) * seg_fu;
          px = -1.;
          py = 2. * v - 1.;
          pz = 2. * u - 1.;
          break;
        case 2: // POSITIVE Y
          v = (y + 1) * seg_sv;
          u = (0.5 * x2 + x + 1) * seg_su;
          px = 2. * u - 1.;
          py = 1.;
          pz = -2. * v + 1.;
          break;
        case 3: // NEGATIVE Y
          v = (y + 1) * seg_sv;
          u = (0.5 * x2 + x + 1) * seg_su;
          px = 2. * u - 1.;
          py = -1.;
          pz = 2. * v - 1.;
          break;
        case 4: // POSITIVE Z
          v = y * seg_fv;
          u = (0.5 * x2 + x) * seg_fu;
          px = 2. * u - 1.;
          py = 2. * v - 1.;
          pz = 1.;
          break;
        case 5: // NEGATIVE Z
          v = y * seg_fv;
          u = (0.5 * x2 + x) * seg_fu;
          px = -2. * u + 1.;
          py = 2. * v - 1.;
          pz = -1.;
          break;
      }
      x_coords.push_back(px);
      y_coords.push_back(py);
      z_coords.push_back(pz);
    }

    result.push_back(x_coords);
    result.push_back(y_coords);
    result.push_back(z_coords);
  }

  sort_coords(result, 0, 0, N);

  return result;
}

template<typename U>
std::vector<U> get_sorted_random_vector(int64_t N, int seed) {
  std::vector<U> randx(N);
  std::random_device rd;
  std::mt19937 gen(rd());
  // TODO Remove random seed when experiments end
  gen.seed(seed);
  // uniform distribution between 0 and 1
  std::uniform_real_distribution<> dis(0, 1);
  for (U& x : randx) x = dis(gen);
  std::sort(randx.begin(), randx.end());
  return randx;
}

template<typename U>
std::vector<U> get_non_negative_vector(int64_t N) {
  std::vector<U> randx(N);
  for (int i = 0; i < N; ++i) {
    randx[i] = i;
  }
  return randx;
}

template<typename U>
void sort_coords(std::vector<std::vector<U>>& coords, int dim, int start, int end) {

  if ((end - start) < 4)
    return;

  /*std::vector<U> coords_max(coords.size());
  std::vector<U> coords_min(coords.size());
  //cacluclate the min and max points in each dimension
  for (size_t d=0; d<coords.size(); ++d) {
    coords_max[d] = coords[d][start];
    coords_min[d] = coords[d][start];
    for (int i=start+1; end; ++i) {
      if (coords[d][i] > coords_max[d])
        coords_max[d] = coords[d][i];
      if (coords[d][i] < coords_min[d])
        coords_min[d] = coords[d][i];
    }
  }

  double radius = 0;
  for (size_t d=0; d<coords.size(); ++d) {
    radius = std::max((coords_max[d] - coords_min[d]) / 2, radius);
  }*/
  std::vector<int> I(end-start);
  std::iota(I.begin(), I.end(), 0);
  std::sort(I.begin(), I.end(),
    [&] (int i, int j) {return coords[dim][i+start] < coords[dim][j+start];});
  

  for (size_t d=0; d<coords.size(); ++d) {
    std::vector<U> temp(end-start);
    for (size_t i=0; i<I.size(); ++i) {
      temp[i] = coords[d][start+I[i]];
    }
    for (size_t i=0; i<I.size(); ++i) {
      coords[d][start+i] = temp[i];
    }
  }

  int mid = std::ceil(double(start+end) / 2.0);
  sort_coords(coords, (dim+1) % coords.size(), start, mid);
  sort_coords(coords, (dim+1) % coords.size(), mid, end);

}

} // namespace hicma
