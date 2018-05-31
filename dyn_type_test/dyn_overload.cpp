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
    int dim;
    std::vector<std::shared_ptr<Node>> data;
    Node(const int i_) : i(i_) {}

    virtual ~Node() {}

    virtual const Node& operator=(std::shared_ptr<Node> A) {}

    virtual const bool is(const int enum_id) const {
      return enum_id == HICMA_NODE;
    }

    virtual const char* is_string() const { return "Node"; }

    virtual std::shared_ptr<Node> add(const Node& B) const {
      std::cout << "Not implemented!!" << std::endl;
    };
};


class Dense : public Node{
  public:
    Dense(const int i) : Node(i) {}

    virtual const bool is(const int enum_id) const override {
      return enum_id == HICMA_DENSE;
    }

    virtual const char* is_string() const override { return "Dense"; }

    std::shared_ptr<Node> add(const Node& B) const override {
      if (B.is(HICMA_DENSE)) {
        std::cout << this->is_string() << " + " << B.is_string();
        std::cout << " works!" << std::endl;
        return std::shared_ptr<Node>(new Dense(this->i + B.i));
      } else {
        std::cout << this->is_string() << " + " << B.is_string();
        std::cout << " is undefined!" << std::endl;
        return std::shared_ptr<Node>(nullptr);
      }
    }
};


class LowRank : public Node{
  public:
    LowRank(const int i) : Node(i) {}

    virtual const bool is(const int enum_id) const override {
      return enum_id == HICMA_LOWRANK;
    }

    virtual const char* is_string() const override { return "LowRank"; }

    std::shared_ptr<Node> add(const Node& B) const override {
      if (B.is(HICMA_LOWRANK)) {
        std::cout << this->is_string() << " + " << B.is_string();
        std::cout << " works!" << std::endl;
        return std::shared_ptr<Node>(new LowRank(this->i + B.i));
      } else {
        std::cout << this->is_string() << " + " << B.is_string();
        std::cout << " is undefined!" << std::endl;
        return std::shared_ptr<Node>(nullptr);
      }
    }
};


class Hierarchical : public Node{
  public:
    int dim;
    std::vector<std::shared_ptr<Node>> data;
    Hierarchical(const int i, const int dim_) : Node(i), dim(dim_) {
      data.resize(dim);
      for (int j=0; j<dim; ++j) {
        if (j%2 == 0) {
          data[j] = std::shared_ptr<Node>(new Dense(j));
        } else {
          if (dim>2) {
            data[j] = std::shared_ptr<Node>(new Hierarchical(j, dim/2));
          } else {
            data[j] = std::shared_ptr<Node>(new LowRank(j));
          }
        }
      }
    }

    virtual const bool is(const int enum_id) const override {
      return enum_id == HICMA_HIERARCHICAL;
    }

    virtual const char* is_string() const override { return "Hierarchical"; }

    Node& operator[](const int i) {
      if (i > dim - 1) throw;
      return *data[i];
    }

    const Node& operator[](const int i) const {
      if (i > dim - 1) throw;
      return *data[i];
    }

    const Node& operator=(const std::shared_ptr<Node> A) override {
      const Node& AR = *A.get();
      if (AR.is(HICMA_HIERARCHICAL)) {
        // This can be avoided if Node has data and dim members!
        const Hierarchical* ap = static_cast<const Hierarchical*>(A.get());
        this->i = AR.i;
        //this->data.swap(ap->data);
        this->data = AR.data;
        return *this;
      } else {
        std::cout << this->is_string() << " = " << AR.is_string();
        std::cout << " not implemented!" << std::endl;
      }
    }

    std::shared_ptr<Node> add(const Node& B) const override {
      if (B.is(HICMA_HIERARCHICAL)) {
        std::cout << this->is_string() << " + " << B.is_string();
        std::cout << " works!" << std::endl;
        const Hierarchical* ap = static_cast<const Hierarchical*>(&B);

        return std::shared_ptr<Node>(
            new Hierarchical(this->i + B.i, this->dim + ap->dim));
      } else if (B.is(HICMA_DENSE)) {
        std::cout << this->is_string() << " + " << B.is_string();
        std::cout << " works!" << std::endl;
        const Hierarchical* ap = static_cast<const Hierarchical*>(&B);

        return std::shared_ptr<Node>(new Dense(this->i + B.i));
      } else {
        std::cout << this->is_string() << " + " << B.is_string();
        std::cout << " is undefined!" << std::endl;
        return std::shared_ptr<Node>(nullptr);
      }
    }
};

std::shared_ptr<Node> operator+(const Node& A, const Node& B) {
  std::cout << "ho" << std::endl;
  return A.add(B);
}

const Node operator+=(Node& A, const Node& B) {
  std::cout << "hi" << std::endl;
  A = A + B;
  return A;
}

int
main(int argc, char** argv) {
  Hierarchical hierarchical(2, 16);

  std::cout << hierarchical[1].i << std::endl;

  hierarchical[1] = hierarchical[3] + hierarchical[3];
  hierarchical[1] += hierarchical[3];

  std::cout << hierarchical[1].i << std::endl;
}

