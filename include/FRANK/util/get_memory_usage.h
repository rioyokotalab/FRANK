#ifndef FRANK_util_get_memory_usage_h
#define FRANK_util_get_memory_usage_h


/**
 * @brief General namespace of the FRANK library
 */
namespace FRANK
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

} // namespace FRANK

#endif // FRANK_util_get_memory_usage_h
