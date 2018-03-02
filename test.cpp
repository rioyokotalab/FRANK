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
  double& operator[](const int i) {
    return data[i];
  }
  Dense() {
    dim[0]=0; dim[1]=0;
  }
};

class Grid : public Node {
public:
  int dim[2];
  std::vector<Node*> data;
  Node* operator[](const int i) {
    return data[i];
  }
  Grid(int m) {
    dim[0] = m;
    dim[1] = 1;
    data.resize(m);
  }
};

int main(int argc, char** argv) {
  Grid x(2);
  x.data.push_back(new Dense);
  (*static_cast<Dense*>(x.data[0])).data.resize(4);
}
