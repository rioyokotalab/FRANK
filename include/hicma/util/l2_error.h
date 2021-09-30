#ifndef hicma_util_l2_error_h
#define hicma_util_l2_error_h


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Matrix;

double l2_error(const Matrix&, const Matrix&);

} // namespace hicma

#endif // hicma_util_l2_error_h
