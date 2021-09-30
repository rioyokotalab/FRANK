#ifndef hicma_util_timer_h
#define hicma_util_timer_h

#include <chrono>
#include <map>
#include <string>
#include <vector>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{
namespace timing
{

class Timer;

// Functions to manipulate global timer state machine
Timer& start(std::string event);

double stop(std::string event);

void clearTimers();

void stopAndPrint(std::string event, int depth = 0);

void printTime(std::string event, int depth = 0);

double getTotalTime(std::string event);

unsigned int getNRuns(std::string event);

// Interface of the Timer class if user wants to create own timers
class Timer {
 public:
  Timer();

  Timer(std::string name, Timer* parent=nullptr);

  void start();

  void start_subtimer(std::string event);

  double stop();

  void clear();

  std::string get_name() const;

  Timer* get_parent() const;

  std::vector<double> get_times() const;

  size_t get_n_runs() const;

  double get_total_time() const;

  const std::map<std::string, double> get_subtimers() const;

  const Timer& operator[](std::string event) const;

  Timer& operator[](std::string event);

  void print_to_depth(int depth) const;

 private:
  using clock = std::chrono::high_resolution_clock;
  typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;
  typedef std::chrono::duration<double> seconds;
  std::string name = "";
  Timer* parent = nullptr;
  bool running = false;
  time_point start_time;
  std::vector<seconds> times;
  seconds total_time = seconds::zero();
  std::map<std::string, Timer> subtimers;

  void print_to_depth(int depth, int at_depth, std::string tag_pre = "") const;
};

} // namespace timing
} // namespace hicma

#endif // hicma_util_timer_h
