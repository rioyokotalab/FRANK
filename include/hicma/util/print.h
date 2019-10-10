#ifndef print_h
#define print_h

#include <string>

namespace hicma {

  extern bool VERBOSE;

  class NodeProxy;

  void printXML(const NodeProxy& A);

  void print(std::string s);

  template<typename T>
  void print(std::string s, T v, bool fixed=true);

}
#endif
