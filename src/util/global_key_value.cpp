#include "hicma/util/global_key_value.h"


namespace hicma {

std::map<std::string, unsigned int> globalKeyValue;

int getGlobalValue(std::string key) {
  if(globalKeyValue.find(key) == globalKeyValue.end()) return -1;
  else return (int)globalKeyValue[key];
}

void setGlobalValue(std::string key, unsigned int value) {
  globalKeyValue[key] = value;
}

} // namespace hicma
