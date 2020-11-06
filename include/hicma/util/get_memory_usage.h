#ifndef hicma_util_get_memory_usage_h
#define hicma_util_get_memory_usage_h


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Matrix;

unsigned long get_memory_usage(const Matrix&, bool include_structure=true);

} // namespace hicma

#endif // hicma_util_get_memory_usage_h
