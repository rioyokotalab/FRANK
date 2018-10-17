#ifndef print_h
#define print_h
#include "print.h"
#include "node.h"
#include "hierarchical.h"
#include "low_rank.h"
#include "dense.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <lapacke.h>

namespace hicma {
  bool VERBOSE = true;                          //!< Print to screen
  static const int stringLength = 24;           //!< Length of formatted string
  static const int decimal = 7;                 //!< Decimal precision
  static const int wait = 100;                  //!< Waiting time between output of different ranks

  void fillXML(const Node& _A, boost::property_tree::ptree& tree) {
    namespace pt = boost::property_tree;
    if (_A.is(HICMA_HIERARCHICAL)) {
      const Hierarchical& A = static_cast<const Hierarchical&>(_A);
      for (int i = 0; i < A.dim[0]; ++i) {
        for (int j = 0; j < A.dim[1]; ++j) {
          pt::ptree el_subtree{};
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
      const LowRank& A = static_cast<const LowRank&>(_A);
      int max_rank = std::min(A.dim[0], A.dim[1]);
      std::vector<double> S(max_rank), U(A.dim[0]*A.dim[1]), Vt(A.dim[1]*A.dim[1]), SU(max_rank - 1);
      LAPACKE_dgesvd(
          LAPACK_ROW_MAJOR, 'A', 'A',
          A.dim[0], A.dim[1], &(Dense(A))[0],
          A.dim[0], &S[0], &U[0], A.dim[0], &Vt[0], A.dim[1], &SU[0]);
      std::string singular_values = std::to_string(S[0]);
      for (int i=1; i<max_rank; ++i)
        singular_values += std::string(",") + std::to_string(S[i]);
      tree.put("<xmlattr>.type", A.type());
      tree.put("<xmlattr>.dim0", A.dim[0]);
      tree.put("<xmlattr>.dim1", A.dim[1]);
      tree.put("<xmlattr>.rank", A.rank);
      tree.put("<xmlattr>.svalues", singular_values);
    } else if (_A.is(HICMA_DENSE)) {
      const Dense& A = static_cast<const Dense&>(_A);
      int max_rank = std::min(A.dim[0], A.dim[1]);
      std::vector<double> S(max_rank), U(A.dim[0]*A.dim[1]), Vt(A.dim[1]*A.dim[1]), SU(max_rank - 1);
      LAPACKE_dgesvd(
          LAPACK_ROW_MAJOR, 'A', 'A',
          A.dim[0], A.dim[1], &(Dense(A))[0],
          A.dim[0], &S[0], &U[0], A.dim[0], &Vt[0], A.dim[1], &SU[0]);
      std::string singular_values = std::to_string(S[0]);
      for (int i=1; i<max_rank; ++i)
        singular_values += std::string(",") + std::to_string(S[i]);
      tree.put("<xmlattr>.type", A.type());
      tree.put("<xmlattr>.dim0", A.dim[0]);
      tree.put("<xmlattr>.dim1", A.dim[1]);
      tree.put("<xmlattr>.svalues", singular_values);
    } else {
      tree.add("Node", "test");
    }
  }

  void printXML(const Node& A) {
    namespace pt = boost::property_tree;
    pt::ptree tree;
    // Write any header info you want here, like a time stamp
    // And then pass pass A into printXML along with the basic ptree
    pt::ptree root_el;
    fillXML(A, root_el);
    tree.add_child("root", root_el);

    pt::xml_writer_settings<std::string> settings(' ', 4);
    write_xml("test.xml", tree, std::locale(), settings);
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
}
#endif
