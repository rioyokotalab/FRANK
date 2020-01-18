#include "hicma/util/timer.h"

#include "hicma/util/print.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace hicma {
namespace timing {

// Global recursive state machine
Timer GlobalTimer;
// Timer into recursive state machine
Timer* current_timer = &GlobalTimer;

void start(std::string event) {
  current_timer->start_subtimer(event);
  current_timer = &(*current_timer)[event];
}

double stop(std::string event) {
  assert(current_timer->get_name() == event);
  double duration = current_timer->stop();
  if (current_timer->get_parent() != nullptr) {
    current_timer = current_timer->get_parent();
  }
  return duration;
}

void clearTimers() {
  GlobalTimer.clear();
}

void stopAndPrint(std::string event, int depth) {
  stop(event);
  printTime(event, depth);
}

void printTime(std::string event, int depth) {
  current_timer->print_to_depth(event, depth);
}


Timer::Timer() {
  name = "";
  parent = nullptr;
  total_duration = seconds::zero();
}

Timer::Timer(std::string name, Timer* parent)
: name(name), parent(parent) {
  total_duration = seconds::zero();
}

void Timer::start() {
  assert(!running);
  running = true;
  start_time = clock::now();
}

void Timer::start_subtimer(std::string event) {
  if (subtimers.find(event) == subtimers.end()) {
    subtimers[event] = Timer(event, this);
  }
  subtimers[event].start();
}

double Timer::stop() {
  assert(running);
  time_point end_time = clock::now();
  running = false;
  seconds duration = end_time - start_time;
  durations.push_back(duration);
  total_duration += duration;
  return duration.count();
}

void Timer::clear() {
  assert(!running);
  assert(parent == nullptr && name.empty());
  total_duration = seconds::zero();
  subtimers.clear();
}

std::string Timer::get_name() const { return name; }

Timer* Timer::get_parent() const { return parent; }

double Timer::get_total_duration() const {
  return total_duration.count();
}

std::vector<double> Timer::get_durations_list() const {
  std::vector<double> durations_list;
  for (const seconds& duration : durations) {
    durations_list.push_back(duration.count());
  }
  return durations_list;
}

size_t Timer::get_number_of_runs() const {
  return durations.size();
}

const std::map<std::string, Timer>& Timer::get_subtimers() const {
  return subtimers;
}

const Timer& Timer::operator[](std::string event) const {
  assert(subtimers.find(event) != subtimers.end());
  return subtimers.at(event);
}

Timer& Timer::operator[](std::string event) {
  assert(subtimers.find(event) != subtimers.end());
  return subtimers[event];
}

void Timer::print_to_depth(
  std::string event, int depth
) const {
  assert(!(*this)[event].running);
  (*this)[event].print_to_depth(depth, 0);
}

void Timer::print_to_depth(int depth, int at_depth, std::string tag_pre) const {
  std::string tag = tag_pre;
  print(at_depth == 0 ? name : tag+"--"+name, total_duration.count());
  if (depth > 0) {
    std::vector<const Timer*> duration_sorted;
    for (const auto& pair : subtimers) {
      duration_sorted.push_back(&pair.second);
    }
    std::sort(
      duration_sorted.begin(), duration_sorted.end(),
      [](const Timer* a, const Timer* b) {
        return a->total_duration > b->total_duration;
      }
    );
    for (const Timer* ptr : duration_sorted) {
      std::string child_tag = tag_pre;
      child_tag += " |";
      ptr->print_to_depth(depth-1, at_depth+1, child_tag);
    }
  }
  if (depth > 0 && subtimers.size() > 0) {
    double subcounter_sum = 0;
    for (const auto& pair : subtimers) {
      subcounter_sum += pair.second.total_duration.count();
    }
    print(
      tag+" |_Subcounters [%]",
      int(std::round(subcounter_sum/total_duration.count()*100))
    );
  }
}

} // namespace timing
} // namespace hicma
