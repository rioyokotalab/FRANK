#include "hicma/util/timer.h"

#include "hicma/util/print.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <string>
#include <vector>


namespace hicma
{
namespace timing
{

// Global recursive state machine
Timer GlobalTimer;
// Timer into recursive state machine
Timer* current_timer = &GlobalTimer;

Timer& start(std::string event) {
  current_timer->start_subtimer(event);
  current_timer = &(*current_timer)[event];
  return *current_timer;
}

// TODO Refactor so this doesn't need event?
double stop([[maybe_unused]] std::string event) {
  assert(current_timer->get_name() == event);
  double duration = current_timer->stop();
  if (current_timer->get_parent() != nullptr) {
    current_timer = current_timer->get_parent();
  }
  return duration;
}

void clearTimers() { GlobalTimer.clear(); }

void stopAndPrint(std::string event, int depth) {
  stop(event);
  printTime(event, depth);
}

void printTime(std::string event, int depth) {
  (*current_timer)[event].print_to_depth(depth);
}

double getTotalTime(std::string event) {
  return (*current_timer)[event].get_total_time();
}

unsigned int getNRuns(std::string event){
  return (*current_timer)[event].get_n_runs();
}

Timer::Timer() {
  name = "";
  parent = nullptr;
  total_time = seconds::zero();
  running = false;
}

Timer::Timer(std::string name, Timer* parent)
: name(name), parent(parent) {
  total_time = seconds::zero();
  running = false;
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
  seconds time = end_time - start_time;
  times.push_back(time);
  total_time += time;
  return time.count();
}

void Timer::clear() {
  assert(!running);
  assert(parent == nullptr && name.empty());
  total_time = seconds::zero();
  subtimers.clear();
}

std::string Timer::get_name() const { return name; }

Timer* Timer::get_parent() const { return parent; }

double Timer::get_total_time() const { return total_time.count(); }

std::vector<double> Timer::get_times() const {
  std::vector<double> times_list;
  for (const seconds& time : times) {
    times_list.push_back(time.count());
  }
  return times_list;
}

size_t Timer::get_n_runs() const { return times.size(); }

const std::map<std::string, double> Timer::get_subtimers() const {
  std::map<std::string, double> subtimer_list;
  for (const auto& pair : subtimers) {
    subtimer_list[pair.first] = pair.second.get_total_time();
  }
  return subtimer_list;
}

const Timer& Timer::operator[](std::string event) const {
  assert(subtimers.find(event) != subtimers.end());
  return subtimers.at(event);
}

Timer& Timer::operator[](std::string event) {
  assert(subtimers.find(event) != subtimers.end());
  return subtimers[event];
}

void Timer::print_to_depth(int depth) const {
  assert(!running);
  print_to_depth(depth, 0);
}

void Timer::print_to_depth(int depth, int at_depth, std::string tag_pre) const {
  std::string tag = tag_pre;
  print(at_depth == 0 ? name : tag+"--"+name, total_time.count());
  if (depth > 0) {
    std::vector<const Timer*> time_sorted;
    for (const auto& pair : subtimers) {
      time_sorted.push_back(&pair.second);
    }
    std::sort(
      time_sorted.begin(), time_sorted.end(),
      [](const Timer* a, const Timer* b) {
        return a->total_time > b->total_time;
      }
    );
    for (const Timer* ptr : time_sorted) {
      std::string child_tag = tag_pre;
      child_tag += " |";
      ptr->print_to_depth(depth-1, at_depth+1, child_tag);
    }
  }
  if (depth > 0 && subtimers.size() > 0) {
    double subcounter_sum = 0;
    for (const auto& pair : subtimers) {
      subcounter_sum += pair.second.total_time.count();
    }
    print(
      tag+" |_Subcounters [%]",
      int(std::round(subcounter_sum/total_time.count()*100))
    );
  }
}

} // namespace timing
} // namespace hicma
