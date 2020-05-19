#include "hicma/util/print.h"
#include "hicma/extension_headers/util.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/low_rank_shared.h"
#include "hicma/classes/node.h"
#include "hicma/classes/no_copy_split.h"
#include "hicma/classes/uniform_hierarchical.h"
#include "hicma/operations/LAPACK.h"
#include "hicma/util/omm_error_handler.h"

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/xml_parser.hpp"
namespace pt = boost::property_tree;
#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>


namespace hicma
{

bool VERBOSE = true;
static const int stringLength = 35; //!< Length of formatted string
static const int decimal = 7; //!< Decimal precision

std::string type(const Node& A) { return type_omm(A); }

define_method(std::string, type_omm, ([[maybe_unused]] const Dense& A)) {
  return "Dense";
}

define_method(std::string, type_omm, ([[maybe_unused]] const LowRank& A)) {
  return "LowRank";
}

define_method(std::string, type_omm, ([[maybe_unused]] const LowRankShared& A)) {
  return "LowRankShared";
}

define_method(std::string, type_omm, ([[maybe_unused]] const Hierarchical& A)) {
  return "Hierarchical";
}

define_method(std::string, type_omm, ([[maybe_unused]] const UniformHierarchical& A)) {
  return "UniformHierarchical";
}

define_method(std::string, type_omm, ([[maybe_unused]] const NoCopySplit& A)) {
  return "NoCopySplit";
}

define_method(std::string, type_omm, (const Node& A)) {
  omm_error_handler("type", {A}, __FILE__, __LINE__);
  std::abort();
}

declare_method(
  void, fillXML_omm,
  (virtual_<const Node&>, pt::ptree&, int64_t, int64_t, int64_t)
)

void fillXML(
  const Node& A, pt::ptree& tree, int64_t i_abs=0, int64_t j_abs=0, int64_t level=0
) {
  fillXML_omm(A, tree, i_abs, j_abs, level);
}

define_method(
  void, fillXML_omm,
  (
    const Hierarchical& A, pt::ptree& tree,
    int64_t i_abs, int64_t j_abs, int64_t level
  )
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      pt::ptree el_subtree{};
      fillXML(A(i, j), el_subtree, i_abs*A.dim[0]+i, j_abs*A.dim[1]+j, level+1);
      std::string el_name = "i" + std::to_string(i) + "j" + std::to_string(j);
      tree.add_child(el_name, el_subtree);
      tree.put(el_name + ".<xmlattr>.type", type(A(i, j)));
    }
  }
  tree.put("<xmlattr>.type", type(A));
  tree.put("<xmlattr>.dim0", A.dim[0]);
  tree.put("<xmlattr>.dim1", A.dim[1]);
  tree.put("<xmlattr>.i_abs", i_abs);
  tree.put("<xmlattr>.j_abs", j_abs);
  tree.put("<xmlattr>.level", level);
}

define_method(
  void, fillXML_omm,
  (const LowRank& A, pt::ptree& tree, int64_t i_abs, int64_t j_abs, int64_t level)
) {
  Dense AD(A);
  Dense S = get_singular_values(AD);
  std::string singular_values = std::to_string(S[0]);
  for (int64_t i=1; i<A.dim[0]; ++i)
    singular_values += std::string(",") + std::to_string(S[i]);
  tree.put("<xmlattr>.type", type(A));
  tree.put("<xmlattr>.dim0", A.dim[0]);
  tree.put("<xmlattr>.dim1", A.dim[1]);
  tree.put("<xmlattr>.i_abs", i_abs);
  tree.put("<xmlattr>.j_abs", j_abs);
  tree.put("<xmlattr>.level", level);
  tree.put("<xmlattr>.rank", A.rank);
  tree.put("<xmlattr>.svalues", singular_values);
}

define_method(
  void, fillXML_omm,
  (const LowRankShared& A, pt::ptree& tree, int64_t i_abs, int64_t j_abs, int64_t level)
) {
  Dense AD(A);
  Dense S = get_singular_values(AD);
  std::string singular_values = std::to_string(S[0]);
  for (int64_t i=1; i<A.dim[0]; ++i)
    singular_values += std::string(",") + std::to_string(S[i]);
  tree.put("<xmlattr>.type", type(A));
  tree.put("<xmlattr>.dim0", A.dim[0]);
  tree.put("<xmlattr>.dim1", A.dim[1]);
  tree.put("<xmlattr>.i_abs", i_abs);
  tree.put("<xmlattr>.j_abs", j_abs);
  tree.put("<xmlattr>.level", level);
  tree.put("<xmlattr>.rank", A.rank);
  tree.put("<xmlattr>.svalues", singular_values);
}

define_method(
  void, fillXML_omm,
  (const Dense& A, pt::ptree& tree, int64_t i_abs, int64_t j_abs, int64_t level)
) {
  Dense A_(A);
  Dense S = get_singular_values(A_);
  std::string singular_values = std::to_string(S[0]);
  for (int64_t i=1; i<A.dim[0]; ++i)
    singular_values += std::string(",") + std::to_string(S[i]);
  tree.put("<xmlattr>.type", type(A));
  tree.put("<xmlattr>.dim0", A.dim[0]);
  tree.put("<xmlattr>.dim1", A.dim[1]);
  tree.put("<xmlattr>.i_abs", i_abs);
  tree.put("<xmlattr>.j_abs", j_abs);
  tree.put("<xmlattr>.level", level);
  tree.put("<xmlattr>.svalues", singular_values);
}

define_method(
  void, fillXML_omm,
  (
    const Node& A, [[maybe_unused]] pt::ptree& tree,
    [[maybe_unused]] int64_t i_abs, [[maybe_unused]] int64_t j_abs,
    [[maybe_unused]] int64_t level
  )
) {
  omm_error_handler("fillXML", {A}, __FILE__, __LINE__);
}

void printXML(const Node& A, std::string filename) {
  pt::ptree tree;
  // Write any header info you want here, like a time stamp
  // And then pass pass A into printXML along with the basic ptree
  pt::ptree root_el;
  fillXML(A, root_el);
  tree.add_child("root", root_el);
  write_xml(filename.c_str(), tree, std::locale());
}

void print(const Node& A) { print_omm(A); }

void print_separation_line() {
  for (int i=0; i<82; ++i) std::cout << "-";
  std::cout << std::endl;
}

define_method(void, print_omm, (const Node& A)) {
  omm_error_handler("print", {A}, __FILE__, __LINE__);
}

define_method(void, print_omm, (const Dense& A)) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      std::cout << std::setw(20) << std::setprecision(15) << A(i, j) << ' ';
    }
    std::cout << std::endl;
  }
  print_separation_line();
}

define_method(void, print_omm, (const LowRank& A)) {
  std::cout << "U : --------------------------------------" << std::endl;
  print(A.U());
  std::cout << "S : --------------------------------------" << std::endl;
  print(A.S());
  std::cout << "V : --------------------------------------" << std::endl;
  print(A.V());
  print_separation_line();
}

define_method(void, print_omm, (const Hierarchical& A)) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      std::cout << type(A(i, j)) << " (" << i << "," << j << ")" << std::endl;
      print(A(i,j));
    }
    std::cout << std::endl;
  }
  print_separation_line();
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
template void print<int64_t>(std::string s, int64_t v, bool fixed=true);
template void print<float>(std::string s, float v, bool fixed=true);
template void print<double>(std::string s, double v, bool fixed=true);

} // namespace hicma
