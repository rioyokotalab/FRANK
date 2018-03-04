#include <boost/any.hpp>
#include <boost/variant.hpp>
#include <string>
#include <iostream>
#include <vector>

class Node {
public:
  int i;
  int j;
  int level;
};

class Dense : public Node {
public:
  std::vector<double> data;
  int dim[2];
  Dense() {
    dim[0]=0; dim[1]=0;
  }
  Dense(int m, int n) {
    dim[0]=m; dim[1]=n; data.resize(m*n);
  }
  const Dense &operator=(const Dense D) {
    data = D.data;
    return *this;
  }
  double& operator[](const int i) {
    return data[i];
  }
  double& operator()(const int i, const int j) {
    assert(i<dim[0] && j<dim[1]);
    return data[i*dim[1]+j];
  }
  const double& operator()(const int i, const int j) const {
    assert(i<dim[0] && j<dim[1]);
    return data[i*dim[1]+j];
  }
};

class Grid : public Node {
public:
  int dim[2];
  std::vector<boost::any> data;
  boost::any& operator[](const int i) {
    return data[i];
  }
  Grid() {
    dim[0] = 0;
    dim[1] = 0;
  }
  Grid(int m) {
    dim[0] = m;
    dim[1] = 1;
    data.resize(m);
  }
};

int main(int argc, char** argv) {
  Grid x(2);
  Dense D(2,2);
  D(0,0) = 3;
  x[0] = D;
  std::cout << boost::any_cast<Dense>(x[0])(0,0) << '\n';
}
