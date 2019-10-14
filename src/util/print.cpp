#ifndef print_h
#define print_h
#include "hicma/util/print.h"
#include "hicma/any.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"

#include <iostream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace hicma {

  bool VERBOSE = true;
  static const int stringLength = 24; //!< Length of formatted string
  static const int decimal = 7; //!< Decimal precision

  void fillXML(const Any& _A, boost::property_tree::ptree& tree) {
    namespace pt = boost::property_tree;
    if (_A.is(HICMA_HIERARCHICAL)) {
      const Hierarchical& A = static_cast<const Hierarchical&>(*_A.ptr);
      for (int i=0; i<A.dim[0]; i++) {
        for (int j=0; j<A.dim[1]; j++) {
          pt::ptree el_subtree;
          fillXML(A(i, j), el_subtree);
          std::string el_name = "i" + std::to_string(i) + "j" + std::to_string(j);
          tree.add_child(el_name, el_subtree);
          tree.put(el_name + ".<xmlattr>.type", A(i, j).type());
        }
      }
      tree.put("<xmlattr>.type", A.type());
      tree.put("<xmlattr>.dim0", A.dim[0]);
      tree.put("<xmlattr>.dim1", A.dim[1]);
    } else if (_A.is(HICMA_LOWRANK)) {
      const LowRank& A = static_cast<const LowRank&>(*_A.ptr);
      Dense S(A.dim[0],1);
      Dense(A).svd(S);
      std::string singular_values = std::to_string(S[0]);
      for (int i=1; i<A.dim[0]; ++i)
        singular_values += std::string(",") + std::to_string(S[i]);
      tree.put("<xmlattr>.type", A.type());
      tree.put("<xmlattr>.dim0", A.dim[0]);
      tree.put("<xmlattr>.dim1", A.dim[1]);
      tree.put("<xmlattr>.rank", A.rank);
      tree.put("<xmlattr>.svalues", singular_values);
    } else if (_A.is(HICMA_DENSE)) {
      const Dense& A = static_cast<const Dense&>(*_A.ptr);
      Dense S(A.dim[0],1);
      Dense(A).svd(S);
      std::string singular_values = std::to_string(S[0]);
      for (int i=1; i<A.dim[0]; ++i)
        singular_values += std::string(",") + std::to_string(S[i]);
      tree.put("<xmlattr>.type", A.type());
      tree.put("<xmlattr>.dim0", A.dim[0]);
      tree.put("<xmlattr>.dim1", A.dim[1]);
      tree.put("<xmlattr>.svalues", singular_values);
    } else {
      tree.add("Node", "test");
    }
  }

  void printXML(const Any& A) {
    namespace pt = boost::property_tree;
    pt::ptree tree;
    // Write any header info you want here, like a time stamp
    // And then pass pass A into printXML along with the basic ptree
    pt::ptree root_el;
    fillXML(A, root_el);
    tree.add_child("root", root_el);
    pt::xml_writer_settings<std::string> settings(' ', 4);
    write_xml("matrix.xml", tree, std::locale(), settings);
  }

  void print(std::string s) {
    if (!VERBOSE) return;
    s += " ";
    std::cout << "--- " << std::setw(stringLength) << std::left
              << std::setfill('-') << s << std::setw(decimal+1) << "-"
              << std::setfill(' ') << std::endl;
  }

  template<typename T>
  void print(std::string s, T v, bool fixed) {
    if (!VERBOSE) return;
    std::cout << std::setw(stringLength) << std::left << s << " : ";
    if(fixed)
      std::cout << std::setprecision(decimal) << std::fixed;
    else
      std::cout << std::setprecision(1) << std::scientific;
    std::cout << v << std::endl;
  }

  template void print<int>(std::string s, int v, bool fixed=true);
  template void print<size_t>(std::string s, size_t v, bool fixed=true);
  template void print<float>(std::string s, float v, bool fixed=true);
  template void print<double>(std::string s, double v, bool fixed=true);

  void print_undefined(std::string func, std::string A_type, std::string B_type, std::string C_type, std::string D_type) {
    std::cerr << D_type << "." << func << "(" << A_type << "," << B_type <<"," <<C_type << ") undefined." << std::endl;
  }

  void print_undefined(std::string func, std::string A_type, std::string B_type, std::string C_type) {
    std::cerr << C_type << "." << func << "(" << A_type << "," << B_type << ") undefined." << std::endl;
  }

  void print_undefined(std::string func, std::string A_type, std::string B_type) {
    std::cerr << B_type << "." << func << "(" << A_type << ") undefined." << std::endl;
  }
}
#endif
