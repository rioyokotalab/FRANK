#ifndef print_h
#define print_h

#include <boost/property_tree/ptree.hpp>

namespace hicma {

  extern bool VERBOSE;

  class Any;

  void printXML(const Any& A);

  void fillXML(const Any& A, boost::property_tree::ptree tree);

  void print(std::string s);

  template<typename T>
  void print(std::string s, T v, bool fixed=true);

  void print_undefined(std::string func, std::string A_type, std::string B_type, std::string C_type);

  void print_undefined(std::string func, std::string A_type, std::string B_type);
}
#endif
