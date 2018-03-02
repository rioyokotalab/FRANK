#ifndef dense_h
#define dense_h
#include <cassert>
#include "node.h"
#include <vector>

namespace hicma {
  class Dense : public Node {
  public:
    std::vector<double> data;
    int dim[2];

    Dense() {
      dim[0]=0; dim[1]=0;
    }

    Dense(int i, int j) {
      dim[0]=i; dim[1]=j;
      data.resize(dim[0]*dim[1]);
    }

    inline double& operator[](int i) {
      return data[i];
    }

    inline double& operator()(int i, int j) {
      assert(i<dim[0] && j<dim[1]);
      return data[i*dim[1]+j];
    }

    const Dense operator+=(const Dense& D) {
      for (int i=0; i<dim[0]*dim[1]; i++)
        this->data[i] += D.data[i];
      return *this;
    }

    Dense operator+(const Dense& D) const {
      return Dense(*this) += D;
    }

    Dense operator*(const Dense& D) const {
      assert(dim[1] == D.dim[0]);
      Dense D2(dim[0],dim[1]);
      for (int i=0; i<dim[0]; i++) {
        for (int j=0; j<dim[1]; j++) {
          double tmp=0;
          for (int k=0; k<D.dim[1]; k++) {
            tmp += data[i*D.dim[1]+k] * D.data[k*dim[1]+j];
          }
          D2.data[i*dim[1]+j]=tmp;
        }
      }
      return D2;
    }

    void resize(int i, int j) {
      dim[0]=i; dim[1]=j;
      data.resize(dim[0]*dim[1]);
    }

    void print() {
      for (int i=0; i<dim[0]; i++) {
        for (int j=0; j<dim[1]; j++) {
          std::cout << data[dim[0]*i+j] << ' ';
        }
        std::cout << std::endl;
      }
    }
  };
}
#endif
