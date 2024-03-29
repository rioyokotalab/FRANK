#include "FRANK/util/print.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/empty.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/operations/LAPACK.h"
#include "FRANK/operations/misc.h"
#include "FRANK/util/omm_error_handler.h"

#include "nlohmann/json.hpp"
#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>


namespace FRANK
{

bool VERBOSE = true;
static const int stringLength = 35; //!< Length of formatted string
static const int decimal = 7; //!< Decimal precision

declare_method(std::string, type_omm, (virtual_<const Matrix&>))

std::string type(const Matrix& A) { return type_omm(A); }

define_method(std::string, type_omm, (const Dense&)) {
  return "Dense";
}

define_method(std::string, type_omm, (const Empty&)) {
  return "Empty";
}

define_method(std::string, type_omm, (const LowRank&)) {
  return "LowRank";
}

define_method(std::string, type_omm, (const Hierarchical&)) {
  return "Hierarchical";
}

define_method(std::string, type_omm, (const Matrix&)) {
  return "Matrix";
}

declare_method(
  void, to_json_omm,
  (virtual_<const Matrix&>, nlohmann::json&, const int64_t, const int64_t, const int64_t)
)

void to_json(
  nlohmann::json& json, const Matrix& A,
  const int64_t i_abs=0, const int64_t j_abs=0, const int64_t level=0
) {
  json["type"] = type(A);
  json["dim"] = {get_n_rows(A), get_n_cols(A)};
  to_json_omm(A, json, i_abs, j_abs, level);
}

define_method(
  void, to_json_omm,
  (
    const Hierarchical& A, nlohmann::json& json,
    const int64_t i_abs, const int64_t j_abs, const int64_t level
  )
) {
  json["abs_pos"] = {i_abs, j_abs};
  json["level"] = level;
  json["children"] = {};
  for (int64_t i=0; i<A.dim[0]; i++) {
    std::vector<nlohmann::json> row(A.dim[1]);
    for (int64_t j=0; j<A.dim[1]; j++) {
      to_json(row[j], A(i, j), i_abs*A.dim[0]+i, j_abs*A.dim[1]+j, level+1);
      // std::string el_name = "i" + std::to_string(i) + "j" + std::to_string(j);
      // tree.add_child(el_name, el_subtree);
      // tree.put(el_name + ".<xmlattr>.type", type(A(i, j)));
    }
    json["children"].push_back(row);
  }
}

define_method(
  void, to_json_omm,
  (
    const LowRank& A, nlohmann::json& json,
    const int64_t i_abs, const int64_t j_abs, const int64_t level
  )
) {
  json["abs_pos"] = {i_abs, j_abs};
  json["level"] = level;
  Dense AD(A);
  json["svalues"] = get_singular_values(AD);
  json["rank"] = A.rank;
}

define_method(
  void, to_json_omm,
  (
    const Dense& A, nlohmann::json& json,
    const int64_t i_abs, const int64_t j_abs, const int64_t level
  )
) {
  json["abs_pos"] = {i_abs, j_abs};
  json["level"] = level;
  Dense A_(A);
  json["svalues"] = get_singular_values(A_);
}

define_method(
  void, to_json_omm,
  (const Matrix& A, nlohmann::json&, const int64_t, const int64_t, const int64_t)
) {
  omm_error_handler("to_json", {A}, __FILE__, __LINE__);
}

void write_JSON(const Matrix& A, const std::string filename) {
  nlohmann::json json(A);
  std::ofstream out_file(filename);
  out_file << json;
}

declare_method(void, print_omm, (virtual_<const Matrix&>))

void print(const Matrix& A) {
  if (!VERBOSE) return;
  print_omm(A);
}

void print_separation_line() {
  for (int i=0; i<82; ++i) std::cout << "-";
  std::cout << std::endl;
}

define_method(void, print_omm, (const Matrix& A)) {
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
  print(A.U);
  std::cout << "S : --------------------------------------" << std::endl;
  print(A.S);
  std::cout << "V : --------------------------------------" << std::endl;
  print(A.V);
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
void print(const std::string s, const T v, const bool fixed) {
  if (!VERBOSE) return;
  std::cout << std::setw(stringLength) << std::left << s << " : ";
  if(fixed)
    std::cout << std::setprecision(decimal) << std::fixed;
  else
    std::cout << std::setprecision(1) << std::scientific;
  std::cout << v << std::endl;
}

template void print<int>(const std::string s, const int v, const bool fixed=true);
template void print<int64_t>(const std::string s, const int64_t v, const bool fixed=true);
template void print<float>(const std::string s, const float v, const bool fixed=true);
template void print<double>(const std::string s, const double v, const bool fixed=true);

} // namespace FRANK
