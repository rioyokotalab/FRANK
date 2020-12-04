#ifndef hicma_util_global_key_value_h
#define hicma_util_global_key_value_h

#include <string>

namespace hicma {

std::string getGlobalValue(std::string key);

void setGlobalValue(std::string key, std::string value);

} // namespace hicma

#endif // hicma_util_global_key_value_h
