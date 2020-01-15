#include "hicma/util/print.h"
#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/low_rank_shared.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/operations/LAPACK/svd.h"

#include <iostream>
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
namespace pt = boost::property_tree;
#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma {

  bool VERBOSE = true;
  static const int stringLength = 30; //!< Length of formatted string
  static const int decimal = 7; //!< Decimal precision

  MULTI_METHOD(
    fillXML_omm, void,
    const virtual_<Node>&, boost::property_tree::ptree& tree
  );

  void fillXML(const Node& A, boost::property_tree::ptree& tree) {
    fillXML_omm(A, tree);
  }

  BEGIN_SPECIALIZATION(
    fillXML_omm, void,
    const Hierarchical& A, boost::property_tree::ptree& tree
  ) {
    for (int i=0; i<A.dim[0]; i++) {
      for (int j=0; j<A.dim[1]; j++) {
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
    tree.put("<xmlattr>.i_abs", A.i_abs);
    tree.put("<xmlattr>.j_abs", A.j_abs);
    tree.put("<xmlattr>.level", A.level);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(
    fillXML_omm, void,
    const LowRank& A, boost::property_tree::ptree& tree
  ) {
    Dense AD(A);
    Dense S = get_singular_values(AD);
    std::string singular_values = std::to_string(S[0]);
    for (int i=1; i<A.dim[0]; ++i)
      singular_values += std::string(",") + std::to_string(S[i]);
    tree.put("<xmlattr>.type", A.type());
    tree.put("<xmlattr>.dim0", A.dim[0]);
    tree.put("<xmlattr>.dim1", A.dim[1]);
    tree.put("<xmlattr>.i_abs", A.i_abs);
    tree.put("<xmlattr>.j_abs", A.j_abs);
    tree.put("<xmlattr>.level", A.level);
    tree.put("<xmlattr>.rank", A.rank);
    tree.put("<xmlattr>.svalues", singular_values);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(
    fillXML_omm, void,
    const LowRankShared& A, boost::property_tree::ptree& tree
  ) {
    Dense AD(A);
    Dense S = get_singular_values(AD);
    std::string singular_values = std::to_string(S[0]);
    for (int i=1; i<A.dim[0]; ++i)
      singular_values += std::string(",") + std::to_string(S[i]);
    tree.put("<xmlattr>.type", A.type());
    tree.put("<xmlattr>.dim0", A.dim[0]);
    tree.put("<xmlattr>.dim1", A.dim[1]);
    tree.put("<xmlattr>.i_abs", A.i_abs);
    tree.put("<xmlattr>.j_abs", A.j_abs);
    tree.put("<xmlattr>.level", A.level);
    tree.put("<xmlattr>.rank", A.rank);
    tree.put("<xmlattr>.svalues", singular_values);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(
    fillXML_omm, void,
    const Dense& A, boost::property_tree::ptree& tree
  ) {
    Dense A_(A);
    Dense S = get_singular_values(A_);
    std::string singular_values = std::to_string(S[0]);
    for (int i=1; i<A.dim[0]; ++i)
      singular_values += std::string(",") + std::to_string(S[i]);
    tree.put("<xmlattr>.type", A.type());
    tree.put("<xmlattr>.dim0", A.dim[0]);
    tree.put("<xmlattr>.dim1", A.dim[1]);
    tree.put("<xmlattr>.i_abs", A.i_abs);
    tree.put("<xmlattr>.j_abs", A.j_abs);
    tree.put("<xmlattr>.level", A.level);
    tree.put("<xmlattr>.svalues", singular_values);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(
    fillXML_omm, void,
    const Node& A, boost::property_tree::ptree& tree
  ) {
    std::cerr << "WARNING: XML output not defined for " << A.type() << "!" << std::endl;
  } END_SPECIALIZATION;

  void printXML(const Node& A, std::string filename) {
    namespace pt = boost::property_tree;
    pt::ptree tree;
    // Write any header info you want here, like a time stamp
    // And then pass pass A into printXML along with the basic ptree
    pt::ptree root_el;
    fillXML(A, root_el);
    tree.add_child("root", root_el);
    write_xml(filename.c_str(), tree, std::locale());
  }

  void print(const Node& A) {
    print_omm(A);
  }

  void print_separation_line() {
    for (int i=0; i<82; ++i) std::cout << "-";
    std::cout << std::endl;
  }

  BEGIN_SPECIALIZATION(
    print_omm, void,
    const Node& A
  ) {
    std::cout << "Uninitialized (Node)" << std::endl;
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(
    print_omm, void,
    const Dense& A
  ) {
    for (int i=0; i<A.dim[0]; i++) {
      for (int j=0; j<A.dim[1]; j++) {
        std::cout << std::setw(20) << std::setprecision(15) << A(i, j) << ' ';
      }
      std::cout << std::endl;
    }
    print_separation_line();
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(
    print_omm, void,
    const LowRank& A
  ) {
    std::cout << "U : --------------------------------------" << std::endl;
    print(A.U());
    std::cout << "S : --------------------------------------" << std::endl;
    print(A.S());
    std::cout << "V : --------------------------------------" << std::endl;
    print(A.V());
    print_separation_line();
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(
    print_omm, void,
    const Hierarchical& A
  ) {
    for (int i=0; i<A.dim[0]; i++) {
      for (int j=0; j<A.dim[1]; j++) {
        std::cout << A(i, j).type() << " (" << i << "," << j << ")" << std::endl;
        print(A(i,j));
      }
      std::cout << std::endl;
    }
    print_separation_line();
  } END_SPECIALIZATION;

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

} // namespace hicma
