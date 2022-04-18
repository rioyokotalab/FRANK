#ifndef hicma_util_get_memory_usage_h
#define hicma_util_get_memory_usage_h


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Matrix;

/**
 * @brief Get the size of a `Matrix` object (in bytes)
 * 
 * Returns the number of bytes used to represent a `Matrix`
 * object in memory.
 * 
 * @param include_structure ff `true` the memory used to store the strucure
 * of the `Matrix` is included in the result (e.g. row_num, col_num, etc.).
 * @return unsigned long size of the `Matrix` in bytes
 */
unsigned long get_memory_usage(const Matrix&, const bool include_structure=true);

} // namespace hicma

#endif // hicma_util_get_memory_usage_h
