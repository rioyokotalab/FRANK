#ifndef hicma_util_counter_h
#define hicma_util_counter_h

#include <map>
#include <string>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma {

extern std::map<std::string,int> globalCounter;

int getCounter(std::string event);

void resetCounter(std::string event);

void updateCounter(std::string event, int d);

void printCounter(std::string event);

} // namespace hicma

#endif // hicma_util_counter_h
