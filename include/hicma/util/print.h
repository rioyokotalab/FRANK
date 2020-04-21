#ifndef print_h
#define print_h

#include <string>

namespace hicma {

  extern bool VERBOSE;

  class Node;

  void printXML(const Node& A, std::string filename = "matrix.xml");

  void print(std::string s);

  template<typename T>
  void print(std::string s, T v, bool fixed=true);

}
#endif
