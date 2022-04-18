#ifndef hicma_util_global_key_value_h
#define hicma_util_global_key_value_h

#include <string>

namespace hicma {

/**
 * @brief Get the global value corresponding to a certain key
 * 
 * If the key is not set within hicam, it falls back to environement
 * variables.
 * 
 * @param key the identifier for the global value
 * @return std::string the value corresponding to the identifier,
 * empty if the key is not found
 */
std::string getGlobalValue(const std::string key);

/**
 * @brief Specify a global key and value pair
 * 
 * @param key the identifier
 * @param value the corresponding value
 */
void setGlobalValue(const std::string key, const std::string value);

} // namespace hicma

#endif // hicma_util_global_key_value_h
