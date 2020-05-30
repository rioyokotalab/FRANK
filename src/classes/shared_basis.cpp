#include "hicma/classes/shared_basis.h"

#include "hicma/classes/dense.h"

#include <memory>


namespace hicma
{

SharedBasis::SharedBasis(const SharedBasis& A)
: representation(std::make_shared<Dense>(A)) {}

SharedBasis& SharedBasis::operator=(const SharedBasis& A) {
  representation = std::make_shared<Dense>(A);
  return *this;
}

SharedBasis::SharedBasis(std::shared_ptr<Dense> A) : representation(A) {}

SharedBasis SharedBasis::share() const { return SharedBasis(representation); }

std::shared_ptr<Dense> SharedBasis::get_ptr() const { return representation; }

} // namespace hicma
