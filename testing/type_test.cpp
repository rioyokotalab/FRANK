#include <memory>
#include <string>
#include <vector>
#include <cstdio>
#include <cassert>

enum {
  NODE,
  DENSE,
  LOWRANK,
  HIERARCHICAL
};

class Node;

class Block {
  public:
    Block() = default;

    Block(const Block& ref);
    Block(const Node& ref);

    ~Block() = default;

    const Block& operator=(const Block& ref);

    const bool is(const int i) const;
    const char* is_string() const;

    const void print() const;

    Block operator+(const Block& rhs);

    const Block& operator[](const int i) const;
    Block& operator[](const int i);

    std::unique_ptr<Node> ptr;
};

class Node {
  public:
    int i = -1;
    virtual const char* is_string() const { return "Node"; }
    virtual const bool is(const int i) const { return i == NODE; }

    virtual const void print() const {
      printf("%s\n\t%i\n", is_string(), i);
    }

    Node() = default;
    Node(int i) : i(i) {}

    virtual ~Node() = default;

    virtual Node* clone() const { return new Node(); }

    virtual Block operator+(const Node& rhs) const {
      return Node(i + rhs.i);
    }
};

class Dense : public Node {
  public:
    int dim = 2;
    std::vector<float> data;
    const char* is_string() const override { return "Dense"; }
    virtual const bool is(const int i) const { return i == DENSE; }

    const void print() const override {
      printf("%s\t%i\n", is_string(), dim);
        printf("\t");
      for (int i = 0; i < dim; ++i) {
        printf("%f ", data[i]);
      }
      printf("\n");
    }
    Dense() = default;

    Dense(const int dim) : dim(dim) {
      data.resize(dim);
    }

    Dense(const int dim, const std::vector<float> data) : dim(dim), data(data) {}

    Dense* clone() const override { return new Dense(*this); }

    Block operator+(const Node& rhs) const override{
      if (rhs.is(DENSE)) {
        assert(dim == static_cast<const Dense&>(rhs).dim);
        std::vector<float> sum;
        sum.resize(dim);
        for (int i = 0; i < dim; ++i) {
          sum[i] = data[i] + static_cast<const Dense&>(rhs).data[i];
        }
        return Dense(dim, sum);
      } else {
        printf(
          "%s + %s not implemented!\n", is_string(), rhs.is_string());
        return Dense(0);
      }
    }
};

class LowRank : public Node {
  public:
    int i = 1;
    Dense U, S, V;
    const char* is_string() const override { return "Low Rank"; }
    virtual const bool is(const int i) const { return i == LOWRANK; }

    const void print() const override {
      printf("%s\n\t%i\n", is_string(), i);
    }
};

class Hierarchical : public Node {
  public:
    int dim = 0;
    std::vector<Block> data;
    const char* is_string() const override { return "Hierarchical"; }
    virtual const bool is(const int i) const { return i == HIERARCHICAL; }

    const void print() const override {
      printf("\n____%s (dim=%i)____\n", is_string(), dim);
      for (int i=0; i < dim; ++i) data[i].print();
      printf("____\n");
    }

    Hierarchical() = default;

    Hierarchical(int dim) : dim(dim) {
      data.resize(dim);
    }

    Hierarchical* clone() const override { return new Hierarchical(*this); }

    const Block& operator[](const int i) const { return data[i]; }
    Block& operator[](const int i) { return data[i]; }
};

// BLOCK IMPLEMENTATION
Block::Block(const Block& ref) : ptr(ref.ptr->clone()) {}
Block::Block(const Node& ref) : ptr(ref.clone()) {}

const Block& Block::operator=(const Block& ref) {
  ptr.reset(ref.ptr->clone());
  return *this;
}

const char* Block::is_string() const { return ptr->is_string(); }
const bool Block::is(const int i) const { return ptr->is(i); }

const void Block::print() const { return ptr->print(); }

Block Block::operator+(const Block& rhs) {
  return *ptr + *rhs.ptr;
}

const Block& Block::operator[](const int i) const {
  if (is(HIERARCHICAL)) {
    return static_cast<const Hierarchical&>(*ptr)[i];
  } else return *this;
}
Block& Block::operator[](const int i) {
  if (is(HIERARCHICAL)) {
    return static_cast<Hierarchical&>(*ptr)[i];
  } else return *this;
}
//\BLOCK IMPLEMENTATION


int main(int argc, char** argv) {
  printf("\n__________________________________________________\n");
  Hierarchical H(4);
  H[0] = Dense(4, std::vector<float>{0, 1, 2, 3});
  H[1] = Dense(4, std::vector<float>{0, 1, 2, 3});
  H[2] = H[0] + H[1];
  H[2].print();

  Hierarchical H2(3);
  H2[0] = Dense(4, std::vector<float>{0, 1, 2, 3});
  H2[1] = Dense(4, std::vector<float>{0, 1, 2, 3});
  H2[2] = H2[0] + H2[1];
  // H2[2].print();
  H[3] = H2;
  H2[0] = Dense(4, std::vector<float>{4, 5, 6, 7});
  H[3][0].print();
  H2[0].print();
}
