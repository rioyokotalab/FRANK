#include "hicma/extension_headers/util.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"


namespace hicma
{

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
