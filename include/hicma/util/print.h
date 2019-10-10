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

  void print_undefined(std::string func, std::string A_type, std::string B_type, std::string C_type, std::string D_type);

  void print_undefined(std::string func, std::string A_type, std::string B_type, std::string C_type);

  void print_undefined(std::string func, std::string A_type, std::string B_type);
}
#endif
