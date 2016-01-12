#ifndef __TIMER_H__
#define __TIMER_H__

#include "util.h"
#define MAX_NAME_LENGTH 53

/**
 * \defgroup timer Timing and cost measurement
 * @{
 *//**
 * \brief local process walltime measurement
 */
void CTF_set_main_args(int argc, const char * const * argv);

/**
 * \defgroup timer Timing and cost measurement
 * @{
 *//**
 * \brief local process walltime measurement
 */
class CTF_Timer{
  public:
    char const * timer_name;
    int index;
    int exited;
    int original;
  
  public:
    CTF_Timer(char const * name);
    ~CTF_Timer();
    void stop();
    void start();
    void exit();
    
};

    
class CTF_Function_timer{
  public:
    char name[MAX_NAME_LENGTH];
    double start_time;
    double start_excl_time;
    double acc_time;
    double acc_excl_time;
    int calls;

    double total_time;
    double total_excl_time;
    int total_calls;

  public: 
    CTF_Function_timer(char const * name_, 
                   double const start_time_,
                   double const start_excl_time_);
    void compute_totals(MPI_Comm comm);
    bool operator<(CTF_Function_timer const & w) const ;
    void print(FILE *         output, 
               MPI_Comm const comm, 
               int const      rank,
               int const      np);
};

/**
 * \brief epoch during which to measure timers
 */
class CTF_Timer_epoch{
  private:
    CTF_Timer * tmr_inner;
    CTF_Timer * tmr_outer;
    std::vector<CTF_Function_timer> saved_function_timers;
    double save_excl_time;
  public:

  public:
    char const * name;
    //create epoch called name
    CTF_Timer_epoch(char const * name_);
    
    CTF_Timer_epoch(){
      saved_function_timers.clear();
    }


    //clears timers and begins epoch
    void begin();

    //prints timers and clears them
    void end();
};

void CTF_set_context(MPI_Comm ctxt);

#endif

