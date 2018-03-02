#include <vector>
#include <iostream>

class Node {
public:
  virtual Node operator * (const Node *);
};

class Dense : public Node {
public:
  int i;
  Dense(int in) {
    i = in;
  }

  Dense operator * (const Dense & other) {
    return other;
  }
};

class Grid : public Node {
public:
  int i;
  Grid(int in) {
    i = in;
  }
  Dense operator * (const Dense & other) {
    this->i += other.i;
    return other;
  }
  Grid operator * (const Grid & other) {
    this->i -= other.i;
    return *this;
  }
};

int main(int argc, char** argv) {
  Grid x(0);
  Dense y(1);
  Grid z(1);
  x * y;
  x * z;
  x * z;
  std::cout << x.i << std::endl;
}
