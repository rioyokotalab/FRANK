#ifndef hicma_util_print_h
#define hicma_util_print_h

#include <string>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Matrix;

extern bool VERBOSE;

/**
 * @brief Get the type of a `Matrix`
 * 
 * Returns the name of the type as a string.
 * I.e. 'Empty', 'Dense', 'LowRank', 'Hierarichical' and 'Matrix'
 * 
 * @return std::string type of the `Matrix`
 */
std::string type(const Matrix&);

/**
 * @brief Stores a `Matrix` in JSON-format
 * 
 * Writes the matrix to a JSON-file in row-major-order
 * under the specified filename.
 * 
 * @param A the matrix to store
 * @param filename  outputfile location
 */
void write_JSON(const Matrix& A, const std::string filename = "matrix.json");

/**
 * @brief Prints the contents of a `Matrix`
 * 
 * Prints the values stored inside a `Matrix`
 * in row-major order.
 * Does not to anything if `VERBOSE` is set to `false`.
 * 
 */
void print(const Matrix&);

/**
 * @brief Prints a string in a preset-format.
 * 
 * This is mainly used for structuring the output of time measurements.
 * Does not to anything if `VERBOSE` is set to `false`. 
 *
 * @param s string to be printed
 */
void print(std::string s);

/**
 * @brief Prints a string + value pair in a preset-format
 * 
 * This is mainly used for structuring the output of time measurements.
 * Does not to anything if `VERBOSE` is set to `false`.
 * 
 * @tparam T datatype of the value
 * @param s string to be printed
 * @param v value to be printed
 * @param fixed if `true` the value is printed in decimal format, 
 * otherwise scientific notation is used
 */
template<typename T>
void print(const std::string s, const T v, const bool fixed=true);

}

#endif // hicma_util_print_h
