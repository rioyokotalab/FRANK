#include <stdlib.h>
#include <vector>
#include <memory>
#include <iostream>

class Node {
  public:
    int i;
    Node(const int i_) : i(i_) {}

    virtual const char* is() const { return "Node"; }

    virtual Node add(const Node& A) {
      std::cout << "Not implemented!!" << std::endl;
    };
};


class Dense : public Node{
  public:
    Dense(const int i) : Node(i) {}

    virtual const char* is() const override { return "Dense"; }

    Node add(const Node& A) override {
      if (A.is() == "Dense") {
        std::cout << this->is() << " + " << A.is() << " works!" << std::endl;
        Dense temp(this->i + A.i);
        return temp;
      } else {
        std::cout << this->is() << " + " << A.is();
        std::cout << " is undefined!" << std::endl;
        return *this;
      }
    }
};


class LowRank : public Node{
  public:
    LowRank(const int i) : Node(i) {}

    virtual const char* is() const override { return "LowRank"; }

    Node add(const Node& A) override {
      if (A.is() == "LowRank") {
        std::cout << this->is() << " + " << A.is() << " works!" << std::endl;
        LowRank temp(this->i + A.i);
        return temp;
      } else {
        std::cout << this->is() << " + " << A.is();
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

    virtual const char* is() const override { return "Hierarchical"; }

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
      if (B.is() == "Hierarchical") {
        std::cout << this->is() << " + " << B.is() << " works!" << std::endl;
        const Hierarchical* ap = static_cast<const Hierarchical*>(&B);

        Hierarchical temp(this->i + B.i, this->dim + ap->dim );

        return temp;
      } else if (B.is() == "Dense") {
        std::cout << this->is() << " + " << B.is() << " works!" << std::endl;
        const Hierarchical* ap = static_cast<const Hierarchical*>(&B);

        Dense temp(this->i + B.i);

        return temp;

      } else {
        std::cout << this->is() << " + " << B.is();
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
  //std::cout << hierarchical.i << std::endl;

}

