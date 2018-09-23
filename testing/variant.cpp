#include <stdlib.h>
#include <variant>
#include <vector>
#include <memory>
#include <iostream>

enum {
  HICMA_NODE,
  HICMA_HIERARCHICAL,
  HICMA_LOWRANK,
  HICMA_DENSE
};

class Hierarchical_;
class LowRank_;
class Dense_;
using Hierarchical = std::unique_ptr<Hierarchical_>;
using LowRank = std::unique_ptr<LowRank_>;
using Dense = std::unique_ptr<Dense_>;

using Block = std::variant<
  std::monostate,
  Hierarchical,
  LowRank,
  Dense
>;

class Dense_ {
  public:
    int i;
    Dense_(int i) : i(i) {}

    Block add(const Block& B) {
      return std::make_unique<Dense_>(i);
    }
};


class LowRank_ {
  public:
    int i;
    LowRank_(int i) : i(i) {}

    Block add(const Block& B) {
      switch(B.index()) {
        case HICMA_LOWRANK:
          return std::make_unique<LowRank_>(i + std::get<LowRank>(B)->i);
        default:
          std::cout << "Error" << std::endl;
          return std::make_unique<LowRank_>(0);
      }
    }
};


class Hierarchical_ {
  public:
    int i;
    int dim;
    Hierarchical_(int i, int dim) : i(i), dim(dim) {
      data.resize(dim);
    }

    Block& operator[](int i) { return data[i]; }
    const Block& operator[](int i) const { return data[i]; }

    Block add(const Block& B) {
      switch(B.index()) {
        case HICMA_DENSE:
          return std::make_unique<Hierarchical_>(
              i + std::get<Hierarchical>(B)->i, dim);
        default:
          std::cout << "Error" << std::endl;
          return std::make_unique<Dense_>(0);
      }
    }
  private:
    std::vector<Block> data;
};

Block operator+(const Block& A, const Block& B) {
  switch(A.index()) {
    case HICMA_HIERARCHICAL:
      return std::get<Hierarchical>(A)->add(B);
    case HICMA_DENSE:
      return std::get<Dense>(A)->add(B);
  }
}

const Block& operator+=(Block& A, const Block& B) {
  A = A + B;
  return A;
}

int
main(int argc, char** argv) {
  Hierarchical_ hierarchical(2, 4);
  hierarchical[0] = Dense_(0);
  std::cout << std::get<Dense>(hierarchical[0]).i << std::endl;
  hierarchical[1] = Dense_(1);
  hierarchical[2] = Dense_(2);
  hierarchical[3] = Dense_(3);

  hierarchical[1] = hierarchical[3] + hierarchical[3];
  hierarchical[1] += hierarchical[3];
}

