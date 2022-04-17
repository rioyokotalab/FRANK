#include "hicma/util/global_key_value.h"

#include <cstdlib>
#include <map>

namespace hicma {

std::map<std::string, std::string> globalKeyValue;

std::string getGlobalValue(const std::string key) {
  if(globalKeyValue.find(key) == globalKeyValue.end()) {
    if(std::getenv(key.c_str()) != nullptr)
      return std::string(std::getenv(key.c_str())); //Fallback to ENV Variable
    else
      return ""; //Return empty string if not found as well
  }
  else return globalKeyValue[key];
}

void setGlobalValue(const std::string key, const std::string value) {
  globalKeyValue[key] = value;
}

} // namespace hicma
