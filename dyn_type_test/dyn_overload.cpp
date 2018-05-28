#include <stdlib.h>
#include <vector>
#include <memory>
#include <iostream>

enum {
  HICMA_NODE,
  HICMA_HIERARCHICAL,
  HICMA_DENSE,
  HICMA_LOWRANK
};

class Node {
  public:
    int i;
    Node(const int i_) : i(i_) {}

    virtual const bool is(const int enum_id) const {
      return enum_id == HICMA_NODE;
    }

    virtual const char* is_string() const { return "Node"; }

    virtual Node add(const Node& B) {
      std::cout << "Not implemented!!" << std::endl;
    };
};


class Dense : public Node{
  public:
    Dense(const int i) : Node(i) {}

    virtual const bool is(const int enum_id) const override {
      return enum_id == HICMA_DENSE;
    }

    virtual const char* is_string() const { return "Dense"; }

    Node add(const Node& B) override {
      if (B.is(HICMA_DENSE)) {
        std::cout << this->is_string() << " + " << B.is_string();
        std::cout  << " works!" << std::endl;
        Dense temp(this->i + B.i);
        return temp;
      } else {
        std::cout << this->is_string() << " + " << B.is_string();
        std::cout << " is undefined!" << std::endl;
        return *this;
      }
    }
};


class LowRank : public Node{
  public:
    LowRank(const int i) : Node(i) {}

    virtual const bool is(const int enum_id) const override {
      return enum_id == HICMA_LOWRANK;
    }

    virtual const char* is_string() const { return "LowRank"; }

    Node add(const Node& B) override {
      if (B.is(HICMA_LOWRANK)) {
        std::cout << this->is_string() << " + " << B.is_string();
        std::cout  << " works!" << std::endl;
        LowRank temp(this->i + B.i);
        return temp;
      } else {
        std::cout << this->is_string() << " + " << B.is_string();
        std::cout << " is undefined!" << std::endl;
        return *this;
      }
    }
};


class Hierarchical : public Node{
  public:
    int dim;
    std::vector<std::unique_ptr<Node>> data;
    Hierarchical(const int i, const int dim_) : Node(i), dim(dim_) {
      data.resize(dim);
      for (int j=0; j<dim; ++j) {
        if (j%2 == 0) {
          data[j] = std::unique_ptr<Node>(new Dense(j));
        } else {
          data[j] = std::unique_ptr<Node>(new LowRank(j));
        }
      }
    }

    virtual const bool is(const int enum_id) const override {
      return enum_id == HICMA_HIERARCHICAL;
    }

    virtual const char* is_string() const { return "Hierarchical"; }

    Node& operator[](const int i) {
      if (i > dim - 1) throw;
      return *data[i];
    }

    const Node& operator[](const int i) const {
      if (i > dim - 1) throw;
      return *data[i];
    }

    //const Node& operator=(const Node& A) {
    //  const Hierarchical* ap = static_cast<const Hierarchical*>(&A);
    //  this->data = ap->data;
    //  return *this;
    //}

    Node add(const Node& B) override {
      if (B.is(HICMA_HIERARCHICAL)) {
        std::cout << this->is_string() << " + " << B.is_string();
        std::cout  << " works!" << std::endl;
        const Hierarchical* ap = static_cast<const Hierarchical*>(&B);

        Hierarchical temp(this->i + B.i, this->dim + ap->dim );

        return temp;
      } else if (B.is(HICMA_DENSE)) {
        std::cout << this->is_string() << " + " << B.is_string();
        std::cout  << " works!" << std::endl;
        const Hierarchical* ap = static_cast<const Hierarchical*>(&B);

        Dense temp(this->i + B.i);

        return temp;
      } else {
        std::cout << this->is_string() << " + " << B.is_string();
        std::cout << " is undefined!" << std::endl;
        return *this;
      }
    }
};

Node operator+(Node& A, const Node& B) {
  return A.add(B);
}

const Node operator+=(Node& A, const Node& B) {
  A = A + B;
  return A;
}

int
main(int argc, char** argv) {
  Hierarchical hierarchical(2, 4);

  std::cout << hierarchical[0].i << std::endl;

  hierarchical[0] = hierarchical[2] + hierarchical[2];
  hierarchical[0] += hierarchical[2];

  std::cout << hierarchical[0].i << std::endl;
}

