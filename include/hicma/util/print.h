#ifndef hicma_util_print_h
#define hicma_util_print_h

#include <string>

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma {

  extern bool VERBOSE;

  class Node;

  void printXML(const Node& A, std::string filename = "matrix.xml");

  void print(const Node&);

  MULTI_METHOD(
    print_omm, void,
    const virtual_<Node>&
  );

  void print(std::string s);

  template<typename T>
  void print(std::string s, T v, bool fixed=true);

}

#endif // hicma_util_print_h
