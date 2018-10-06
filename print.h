#ifndef print_h
#define print_h
#include "mpi_utils.h"
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#include <boost/property_tree/ptree.hpp>

namespace hicma {
  extern bool VERBOSE;                          //!< Print to screen

  class Node;
  void printXML(const Node& A);
  void fillXML(const Node& A, boost::property_tree::ptree tree);

  void print(std::string s);

  template<typename T>
  void print(std::string s, T v, bool fixed=true);

  template<typename T>
  void printMPI(T data);

  template<typename T>
  void printMPI(T data, const int irank);

  template<typename T>
  void printMPI(T * data, const int begin, const int end);

  template<typename T>
  void printMPI(T * data, const int begin, const int end, const int irank);
}
#endif
