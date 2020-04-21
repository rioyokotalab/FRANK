#ifndef counter_h
#define counter_h
#include <map>

namespace hicma {

  extern std::map<std::string,int> globalCounter;

  int getCounter(std::string event);

  void resetCounter(std::string event);

  void updateCounter(std::string event, int d);

  void printCounter(std::string event);
}
#endif
