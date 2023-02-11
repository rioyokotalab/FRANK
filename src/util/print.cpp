#include "hicma/util/print.h"
#include "hicma/extension_headers/util.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/empty.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/operations/LAPACK.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"

#include "nlohmann/json.hpp"
#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>


namespace hicma
{

bool VERBOSE = true;
static const int stringLength = 35; //!< Length of formatted string
static const int decimal = 7; //!< Decimal precision

std::string type(const Matrix& A) { return type_omm(A); }

define_method(std::string, type_omm, (const Dense<float>&)) {
  return "Dense";
}

define_method(std::string, type_omm, (const Dense<double>&)) {
  return "Dense";
}

define_method(std::string, type_omm, (const Empty&)) {
  return "Empty";
}

define_method(std::string, type_omm, (const LowRank<float>&)) {
  return "LowRank";
}

define_method(std::string, type_omm, (const LowRank<double>&)) {
  return "LowRank";
}

define_method(std::string, type_omm, (const Hierarchical<float>&)) {
  return "Hierarchical";
}

define_method(std::string, type_omm, (const Hierarchical<double>&)) {
  return "Hierarchical";
}

define_method(std::string, type_omm, (const Matrix&)) {
  return "Matrix";
}

declare_method(
  void, to_json_omm,
  (virtual_<const Matrix&>, nlohmann::json&, int64_t, int64_t, int64_t)
)

void to_json(
  nlohmann::json& json, const Matrix& A,
  int64_t i_abs=0, int64_t j_abs=0, int64_t level=0
) {
  json["type"] = type(A);
  json["dim"] = {get_n_rows(A), get_n_cols(A)};
  to_json_omm(A, json, i_abs, j_abs, level);
}

template<typename T>
void hierarchical_to_json(const Hierarchical<T>& A, nlohmann::json& json,
  int64_t i_abs, int64_t j_abs, int64_t level) {
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
    const Hierarchical<float>& A, nlohmann::json& json,
    int64_t i_abs, int64_t j_abs, int64_t level
  )
) {
  hierarchical_to_json(A, json, i_abs, j_abs, level);
}

define_method(
  void, to_json_omm,
  (
    const Hierarchical<double>& A, nlohmann::json& json,
    int64_t i_abs, int64_t j_abs, int64_t level
  )
) {
  hierarchical_to_json(A, json, i_abs, j_abs, level);
}

template<typename T>
void low_rank_to_json(const LowRank<T>& A, nlohmann::json& json,
  int64_t i_abs, int64_t j_abs, int64_t level) {
  json["abs_pos"] = {i_abs, j_abs};
  json["level"] = level;
  Dense<double> AD(A);
  json["svalues"] = get_singular_values(AD);
  json["rank"] = A.rank;
}

define_method(
  void, to_json_omm,
  (
    const LowRank<float>& A, nlohmann::json& json,
    int64_t i_abs, int64_t j_abs, int64_t level
  )
) {
  low_rank_to_json(A, json, i_abs, j_abs, level);
}

define_method(
  void, to_json_omm,
  (
    const LowRank<double>& A, nlohmann::json& json,
    int64_t i_abs, int64_t j_abs, int64_t level
  )
) {
  low_rank_to_json(A, json, i_abs, j_abs, level);
}

template<typename T>
void dense_to_json(const Dense<T>& A, nlohmann::json& json,
  int64_t i_abs, int64_t j_abs, int64_t level) {
  json["abs_pos"] = {i_abs, j_abs};
  json["level"] = level;
  Dense<double> A_(A);
  json["svalues"] = get_singular_values(A_);
}

define_method(
  void, to_json_omm,
  (
    const Dense<float>& A, nlohmann::json& json,
    int64_t i_abs, int64_t j_abs, int64_t level
  )
) {
  dense_to_json(A, json, i_abs, j_abs, level);
}

define_method(
  void, to_json_omm,
  (
    const Dense<double>& A, nlohmann::json& json,
    int64_t i_abs, int64_t j_abs, int64_t level
  )
) {
  dense_to_json(A, json, i_abs, j_abs, level);
}

define_method(
  void, to_json_omm,
  (const Matrix& A, nlohmann::json&, int64_t, int64_t, int64_t)
) {
  omm_error_handler("to_json", {A}, __FILE__, __LINE__);
}

void write_JSON(const Matrix& A, std::string filename) {
  nlohmann::json json(A);
  std::ofstream out_file(filename);
  out_file << json;
}

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

template<typename T>
void print_dense(const Dense<T>& A) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      std::cout << std::setw(20) << std::setprecision(15) << A(i, j) << ' ';
    }
    std::cout << std::endl;
  }
  print_separation_line();
}

define_method(void, print_omm, (const Dense<float>& A)) {
  print_dense(A);
}

define_method(void, print_omm, (const Dense<double>& A)) {
  print_dense(A);
}

template<typename T>
void print_low_rank(const LowRank<T>& A) {
  std::cout << "U : --------------------------------------" << std::endl;
  print(A.U);
  std::cout << "S : --------------------------------------" << std::endl;
  print(A.S);
  std::cout << "V : --------------------------------------" << std::endl;
  print(A.V);
  print_separation_line();
}

define_method(void, print_omm, (const LowRank<float>& A)) {
  print_low_rank(A);
}

define_method(void, print_omm, (const LowRank<double>& A)) {
  print_low_rank(A);
}

template<typename T>
void print_hierarchical(const Hierarchical<T>& A) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      std::cout << type(A(i, j)) << " (" << i << "," << j << ")" << std::endl;
      print(A(i,j));
    }
    std::cout << std::endl;
  }
  print_separation_line();
}

define_method(void, print_omm, (const Hierarchical<float>& A)) {
  print_hierarchical(A);
}

define_method(void, print_omm, (const Hierarchical<double>& A)) {
  print_hierarchical(A);
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

define_method(bool, is_double, (const Dense<double>&)) {
  return true;
}

define_method(bool, is_double, (const Dense<float>&)) {
  return false;
}

define_method(bool, is_double, (const LowRank<double>&)) {
  return true;
}

define_method(bool, is_double, (const LowRank<float>&)) {
  return false;
}

define_method(bool, is_double, (const Hierarchical<double>&)) {
  return true;
}

define_method(bool, is_double, (const Matrix& A)) {
  omm_error_handler("is_double", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
