#pragma once

#include <err.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>

/*
what perf does?

TODO:
1. configure which events our counter should measure
2. cal syscall
3. read values directly with __rdpmc(index) and rdtsc() for time, with no kernel involved

start with counting
1. cycles
2. l1 misses
3.

*/

enum Counter
{
  instructions,
};

// template<Counter... counters>
class PerfCounter
{
  public:
  
  PerfCounter()
  {
    memset(&pe, 0, sizeof(pe));               // zero every field, so unused options are "off"
    pe.type = PERF_TYPE_HARDWARE;             // category: generic hardware event
    pe.size = sizeof(pe);                     // struct size, so the kernel knows which version we use
    pe.config = PERF_COUNT_HW_INSTRUCTIONS;   // the specific event: retired instructions
    pe.disabled = 1;                          // create it stopped; we start it explicitly later
    pe.exclude_kernel = 1;                    // dont count instructions executed in the kernel
    pe.exclude_hv = 1;                        // dont count instructions in the hypervisor
    create_counter();
  }
  
  ~PerfCounter()
  {
    close(fd);
  }

  PerfCounter(const PerfCounter&) = delete;
  PerfCounter& operator=(const PerfCounter&) = delete;

  static long perf_event_open(
    struct perf_event_attr *hw_event, // tells what to count 
    pid_t pid,                        // which process
    int cpu,                          // which cpu : -1 = any cpu the thread runs on
    int group_fd,                     // group leader fd: -1 = this event is its own leader
    unsigned long flags               // extra options
  )
  {
    long ret;
    // syscall returns a file descriptor, a handle to hardware counter regs managed by the kernel
    ret = syscall(
      SYS_perf_event_open,
      hw_event,  // tells the kernel what to count
      pid,
      cpu,
      group_fd,
      flags
    );

    return ret;
  }

  void create_counter()
  {
    fd = perf_event_open(&pe, 0, -1, -1, 0);
    
    if (fd == -1)
    {
      err(EXIT_FAILURE, "Error opening leader %llx\n", pe.config);
    }
  }
  
  

  template <typename Func, typename... Args>
  uint64_t measure(Func&& testFunc, Args&&... args)
  {
    if (ioctl(fd, PERF_EVENT_IOC_RESET, 0) == -1)    // set the counter to zero
    {
      err(EXIT_FAILURE, "PERF_EVENT_IOC_RESET");
    }

    if (ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) == -1)   // start counting from here
    {
      err(EXIT_FAILURE, "PERF_EVENT_IOC_ENABLE");
    }

    std::forward<Func>(testFunc)(std::forward<Args>(args)...);

    if (ioctl(fd, PERF_EVENT_IOC_DISABLE, 0) == -1)  // stop counting
    {
      err(EXIT_FAILURE, "PERF_EVENT_IOC_DISABLE");
    }

    uint64_t result;

    if (read(fd, &result, sizeof(result)) != sizeof(result))  // ask the kernel for the value (8 bytes)
    {
      err(EXIT_FAILURE, "read");
    }

    return result;
  }

  private:
    int fd = -1;          // handle to the counter
    perf_event_attr pe;   // configuration: tells the kernel what to count
};

