#ifndef hicma_util_global_key_value_h
#define hicma_util_global_key_value_h

#include <map>
#include <string>


namespace hicma {

int getGlobalValue(std::string key);

void setGlobalValue(std::string key, unsigned int value);

} // namespace hicma

#endif // hicma_util_global_key_value_h
