/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __UTIL_H__
#define __UTIL_H__

#include <string.h>
#include <string>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>
#include <list>
#include <vector>
#include <complex>
#include <unistd.h>
#include <iostream>

typedef int64_t long_int;
volatile static long_int long_int_max = INT64_MAX;

#include "comm.h"
#include "lapack.h"



#if (defined(__X86_64__) || defined(__IA64__) || defined(__amd64__) || \
     defined(__ppc64__) || defined(_ARCH_PPC) || defined(BGQ) || defined(BGP))
#define PRId64 "%ld"
#define PRIu64 "%lu"
#else //if (defined(__i386__))
#define PRId64 "%lld"
#define PRIu64 "%llu"
//#else
//#include <inttypes.h>
#endif

#ifdef PROFILE
#define TAU
#endif
#include "timer.h"
#include "pmpi.h"
#ifndef __APPLE__
#ifndef OMP_OFF
#define USE_OMP
#include "omp.h"
#endif
#endif

#define COUNT_FLOPS

#ifdef COUNT_FLOPS
#define FLOPS_ADD(n) flops_add(n)
#else
#define FLOPS_ADD(n) 
#endif

void flops_add(long_int n);
long_int get_flops();

inline double GET_REAL(double const d) {
  return d;
}
inline  double GET_REAL(std::complex<double> const d) {
  return d.real();
}
//doesn't work with OpenMPI
//volatile static long_int mpi_long_int = MPI_LONG_LONG_INT;

#ifndef ENABLE_ASSERT
#ifdef DEBUG
#define ENABLE_ASSERT 1
#else
#define ENABLE_ASSERT 0
#endif
#endif
#ifdef _SC_PHYS_PAGES
inline
uint64_t getTotalSystemMemory()
{
  uint64_t pages = (uint64_t)sysconf(_SC_PHYS_PAGES);
  uint64_t page_size = (uint64_t)sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
}
#else
inline
uint64_t getTotalSystemMemory()
{
  //Assume system memory is 1 GB
  return ((uint64_t)1)<<30;
}
#endif

#include <execinfo.h>
#include <signal.h>
inline void handler() {
#if (!BGP && !BGQ && !HOPPER)
  int i, size;
  void *array[26];

  // get void*'s for all entries on the stack
  size = backtrace(array, 25);

  // print out all the frames to stderr
  backtrace_symbols(array, size);
  char syscom[256*size];
  for (i=1; i<size; ++i)
  {
    char buf[256];
    char buf2[256];
    int bufsize = 256;
    int sz = readlink("/proc/self/exe", buf, bufsize);
    buf[sz] = '\0';
    sprintf(buf2,"addr2line %p -e %s", array[i], buf);
    if (i==1)
      strcpy(syscom,buf2);
    else
      strcat(syscom,buf2);

  }
  int *iiarr = NULL;
  iiarr[0]++;
  assert(system(syscom)==0);
  printf("%d",iiarr[0]);
#endif
}
#ifndef LIBT_ASSERT
//#if ENABLE_ASSERT
#define LIBT_ASSERT(...)                \
do { if (!(__VA_ARGS__)) handler(); assert(__VA_ARGS__); } while (0)
#else
#define LIBT_ASSERT(...) do {} while(0 && (__VA_ARGS__))
#endif
//#endif

#define ABORT                                   \
  do{                                           \
  handler(); MPI_Abort(MPI_COMM_WORLD, -1); } while(0)

//proper modulus for 'a' in the range of [-b inf]
#ifndef WRAP
#define WRAP(a,b)       ((a + b)%b)
#endif

#ifndef ALIGN_BYTES
#define ALIGN_BYTES     16
#endif

#ifndef MIN
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#ifndef MAX
#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef LOC
#define LOC \
  do { printf("debug:%s:%d ",__FILE__,__LINE__); } while(0)
#endif

#ifndef ERROR
#define ERROR(...) \
do { printf("error:%s:%d ",__FILE__,__LINE__); printf(__VA_ARGS__); printf("\n"); quit(1); } while(0)
#endif

#ifndef WARN
#define WARN(...) \
do { printf("warning: "); printf(__VA_ARGS__); printf("\n"); } while(0)
#endif

#if defined(VERBOSE)
  #ifndef VPRINTF
  #define VPRINTF(i,...) \
    do { if (i<=VERBOSE) { \
      printf("CTF: "__VA_ARGS__); } \
    } while (0)
  #endif
#else
  #ifndef VPRINTF
  #define VPRINTF(...) do { } while (0)
  #endif
#endif


#ifdef DEBUG
  #ifndef DPRINTF
  #define DPRINTF(i,...) \
    do { if (i<=DEBUG) { LOC; printf(__VA_ARGS__); } } while (0)
  #endif
  #ifndef DEBUG_PRINTF
  #define DEBUG_PRINTF(...) \
    do { DPRINTF(5,__VA_ARGS__); } while(0)
  #endif
  #ifndef RANK_PRINTF
  #define RANK_PRINTF(myRank,rank,...) \
    do { if (myRank == rank) { LOC; printf("P[%d]: ",rank); printf(__VA_ARGS__); } } while(0)
  #endif
        #ifndef PRINT_INT
        #define PRINT_INT(var) \
          do {  LOC; printf(#var); printf("=%d\n",var); } while(0)
        #endif
        #ifndef PRINT_DOUBLE
        #define PRINT_DOUBLE(var) \
          do {  LOC; printf(#var); printf("=%lf\n",var); } while(0)
        #endif
#else
  #ifndef DPRINTF
  #define DPRINTF(...) do { } while (0)
  #endif
  #ifndef DEBUG_PRINTF
  #define DEBUG_PRINTF(...) do {} while (0)
  #endif
  #ifndef RANK_PRINTF
  #define RANK_PRINTF(...) do { } while (0)

  #endif
  #ifndef PRINT_INT
  #define PRINT_INT(var)
  #endif
#endif


#ifdef DUMPDEBUG
  #ifndef DUMPDEBUG_PRINTF
  #define DUMPDEBUG_PRINTF(...) \
    do { LOC; printf(__VA_ARGS__); } while(0)
  #endif
#else
  #ifndef DUMPDEBUG_PRINTF
  #define DUMPDEBUG_PRINTF(...)
  #endif
#endif

/*#ifdef TAU
#include <stddef.h>
#include <Profile/Profiler.h>
#define TAU_FSTART(ARG)                                 \
    TAU_PROFILE_TIMER(timer##ARG, #ARG, "", TAU_USER);  \
    TAU_PROFILE_START(timer##ARG)

#define TAU_FSTOP(ARG)                                  \
    TAU_PROFILE_STOP(timer##ARG)

#else*/

#ifdef PROFILE
#define TAU
#endif

#ifdef TAU
#define TAU_FSTART(ARG)                                           \
  do { CTF_Timer t(#ARG); t.start(); } while (0);

#define TAU_FSTOP(ARG)                                            \
  do { CTF_Timer t(#ARG); t.stop(); } while (0);

#define TAU_PROFILE_TIMER(ARG1, ARG2, ARG3, ARG4)                 

#define TAU_PROFILE_INIT(argc, argv)                              \
  CTF_set_main_args(argc, argv);

#define TAU_PROFILE_SET_NODE(ARG)

#define TAU_PROFILE_START(ARG)                                    \
  CTF_Timer __CTF_Timer##ARG(#ARG);

#define TAU_PROFILE_STOP(ARG)                                     \
 __CTF_Timer##ARG.stop();

#define TAU_PROFILE_SET_CONTEXT(ARG)                              \
  if (ARG==0) CTF_set_context(MPI_COMM_WORLD);                    \
  else CTF_set_context((MPI_Comm)ARG);
#endif




#ifndef TAU
#define TAU_PROFILE(NAME,ARG,USER)
#define TAU_PROFILE_TIMER(ARG1, ARG2, ARG3, ARG4)
#define TAU_PROFILER_CREATE(ARG1, ARG2, ARG3, ARG4)
#define TAU_PROFILE_STOP(ARG)
#define TAU_PROFILE_START(ARG)
#define TAU_PROFILE_SET_NODE(ARG)
#define TAU_PROFILE_SET_CONTEXT(ARG)
#define TAU_FSTART(ARG)
#define TAU_FSTOP(ARG)
#endif




#if (defined(COMM_TIME))
#define INIT_IDLE_TIME                  \
  volatile double __idleTime=0.0;       \
  volatile double __idleTimeDelta=0.0;
#define INSTRUMENT_BARRIER(COMM)        do {    \
  __idleTimeDelta = TIME_SEC();                 \
  COMM_BARRIER(COMM);                           \
  __idleTime += TIME_SEC() - __idleTimeDelta;   \
  } while(0)
#define INSTRUMENT_GLOBAL_BARRIER(COMM) do {    \
  __idleTimeDelta = TIME_SEC();                 \
  GLOBAL_BARRIER(COMM);                         \
  __idleTime += TIME_SEC() - __idleTimeDelta;   \
  } while(0)
#define AVG_IDLE_TIME(cdt, p)                                   \
do{                                                             \
  REDUCE((void*)&__idleTime, (void*)&__idleTimeDelta, 1,        \
          MPI_DOUBLE, COMM_OP_SUM, 0, cdt);                  \
  __idleTime = __idleTimeDelta/p;                               \
}while(0)
#define IDLE_TIME_PRINT_ITER(iter)                              \
  do { printf("%lf seconds spent idle per iteration\n",         \
      __idleTime/iter); } while(0)

#else
#define INSTRUMENT_BARRIER(COMM)
#define INSTRUMENT_GLOBAL_BARRIER(COMM)
#define INIT_IDLE_TIME
#define AVG_IDLE_TIME(cdt, p)
#define IDLE_TIME_PRINT_ITER(iter)
#endif

#define TIME(STRING) TAU_PROFILE(STRING, " ", TAU_DEFAULT)

#ifdef COMM_TIME
//ugly and scope limited, but whatever.
#define INIT_COMM_TIME                          \
  volatile double __commTime =0.0, __commTimeDelta;     \
  volatile double __critTime =0.0, __critTimeDelta;

#define COMM_TIME_START()                       \
  do { __commTimeDelta = TIME_SEC(); } while(0)
#define COMM_TIME_END()                         \
  do { __commTime += TIME_SEC() - __commTimeDelta; } while(0)
#define CRIT_TIME_START()                       \
  do {                                          \
    __commTimeDelta = TIME_SEC();               \
    __critTimeDelta = TIME_SEC();               \
  } while(0)
#define CRIT_TIME_END()                         \
  do {                                          \
    __commTime += TIME_SEC() - __commTimeDelta; \
    __critTime += TIME_SEC() - __critTimeDelta; \
  } while(0)
#define COMM_TIME_PRINT()                       \
  do { printf("%lf seconds spent doing communication\n", __commTime); } while(0)
#define COMM_TIME_PRINT_ITER(iter)                              \
  do { printf("%lf seconds spent doing communication per iteration\n", __commTime/iter); } while(0)
#define CRIT_TIME_PRINT_ITER(iter)                              \
  do { printf("%lf seconds spent doing communication along critical path per iteration\n", __critTime/iter); \
  } while(0)
#define AVG_COMM_TIME(cdt, p)                                                           \
do{                                                                                     \
  REDUCE((void*)&__commTime, (void*)&__commTimeDelta, 1, MPI_DOUBLE, COMM_OP_SUM, 0, cdt);           \
  __commTime = __commTimeDelta/p;                                                       \
}while(0)
#define SUM_CRIT_TIME(cdt, p)                                                           \
do{                                                                                     \
  REDUCE((void*)&__critTime, (void*)&__critTimeDelta, 1, MPI_DOUBLE, COMM_OP_SUM, 0, cdt);           \
  __critTime = __critTimeDelta;                                                 \
}while(0)


void __CM(const int     end,
          const CommData *cdt,
          const int     p,
          const int     iter,
          const int     myRank);
#else
#define __CM(...)
#define INIT_COMM_TIME
#define COMM_TIME_START()
#define COMM_TIME_END()
#define COMM_TIME_PRINT()
#define COMM_TIME_PRINT_ITER(iter)
#define AVG_COMM_TIME(cdt, p)
#define CRIT_TIME_START()
#define CRIT_TIME_END()
#define CRIT_TIME_PRINT_ITER(iter)
#define SUM_CRIT_TIME(cdt, p)
#endif

#define MST_ALIGN_BYTES ALIGN_BYTES

  


template<typename dtype>
void transp(int size,  int lda_i, int lda_o,
            const dtype *A, dtype *B);

template<typename dtype>
void coalesce_bwd(dtype         *B,
                  dtype const   *B_aux,
                  int     k,
                  int     n,
                  int     kb);

/* Copies submatrix to submatrix */
template<typename dtype>
void lda_cpy(int nrow,  int ncol,
             int lda_A, int lda_B,
             const dtype *A,        dtype *B);

template<typename dtype>
void lda_cpy(int nrow,  int ncol,
             int lda_A, int lda_B,
             const dtype *A,        dtype *B,
             const dtype a,  const dtype b);

void print_matrix(double const * M, int n, int m);
void print_matrix(double const * M, int n, int m, int lda);

//double util_dabs(double x);

long_int sy_packed_size(const int ndim, const int* len, const int* sym);

long_int packed_size(const int ndim, const int* len, const int* sym);


/*
 * \brief calculates dimensional indices corresponding to a symmetric-packed index
 *        For each symmetric (SH or AS) group of size sg we have
 *          idx = n*(n-1)*...*(n-sg) / d*(d-1)*...
 *        therefore (idx*sg!)^(1/sg) >= n-sg
 *        or similarly in the SY case ... >= n
 */
void calc_idx_arr(int         ndim,
                  int const * lens,
                  int const * sym,
                  long_int    idx,
                  int *       idx_arr);

void factorize(int n, int *nfactor, int **factor);

inline
int gcd(int a, int b){
  if (b==0) return a;
  return gcd(b, a%b);
}

inline
int lcm(int a, int b){
  return a*b/gcd(a,b);
}

/**
 * \brief Copies submatrix to submatrix (column-major)
 * \param[in] nrow number of rows
 * \param[in] ncol number of columns
 * \param[in] lda_A lda along rows for A
 * \param[in] lda_B lda along rows for B
 * \param[in] A matrix to read from
 * \param[in,out] B matrix to write to
 */
template<typename dtype>
void lda_cpy(const int nrow,  const int ncol,
             const int lda_A, const int lda_B,
             const dtype *A,        dtype *B){
  if (lda_A == nrow && lda_B == nrow){
    memcpy(B,A,nrow*ncol*sizeof(dtype));
  } else {
    int i;
    for (i=0; i<ncol; i++){
      memcpy(B+lda_B*i,A+lda_A*i,nrow*sizeof(dtype));
    }
  }
}

/**
 * \brief Copies submatrix to submatrix with scaling (column-major)
 * \param[in] nrow number of rows
 * \param[in] ncol number of columns
 * \param[in] lda_A lda along rows for A
 * \param[in] lda_B lda along rows for B
 * \param[in] A matrix to read from
 * \param[in,out] B matrix to write to
 * \param[in] a factor to scale A
 * \param[in] b factor to scale B
 */
template<typename dtype>
void lda_cpy(const int nrow,  const int ncol,
             const int lda_A, const int lda_B,
             const dtype *A,        dtype *B,
             const dtype a,  const dtype b){
  int i,j;
  if (lda_A == nrow && lda_B == nrow){
    for (j=0; j<nrow*ncol; j++){
      B[j] = B[j]*b + A[j]*a;
    }
  } else {
    for (i=0; i<ncol; i++){
      for (j=0; j<nrow; j++){
        B[lda_B*i + j] = B[lda_B*i + j]*b + A[lda_A*i + j]*a;
      }
    }
  }
}

/**
 * \brief we receive a contiguous buffer kb-by-n B and (k-kb)-by-n B_aux
 * which is the block below.
 * To get a k-by-n buffer, we need to combine this buffer with our original
 * block. Since we are working with column-major ordering we need to interleave
 * the blocks. Thats what this function does.
 * \param[in,out] B the buffer to coalesce into
 * \param[in] B_aux the second buffer to coalesce from
 * \param[in] k the total number of rows
 * \param[in] n the number of columns
 * \param[in] kb the number of rows in a B originally
 */
template<typename dtype>
void coalesce_bwd(dtype         *B,
                  dtype const   *B_aux,
                  int const     k,
                  int const     n,
                  int const     kb){
  int i;
  for (i=n-1; i>=0; i--){
    memcpy(B+i*k+kb, B_aux+i*(k-kb), (k-kb)*sizeof(dtype));
    if (i>0) memcpy(B+i*k, B+i*kb, kb*sizeof(dtype));
  }
}


/* Copies submatrix to submatrix */
template<typename dtype>
void transp(const int size,  const int lda_i, const int lda_o,
            const dtype *A, dtype *B){
  if (lda_i == 1){
    memcpy(B,A,size*sizeof(dtype));
  }
  int i,j,o;
  LIBT_ASSERT(size%lda_o == 0);
  LIBT_ASSERT(lda_o%lda_i == 0);
  for (o=0; o<size/lda_o; o++){
    for (j=0; j<lda_i; j++){
      for (i=0; i<lda_o/lda_i; i++){
        B[o*lda_o + j*lda_o/lda_i + i] = A[o*lda_o+j+i*lda_i];
      }
    }
  }
}

template<typename dtype>
dtype get_zero(){
  ABORT;
}
template<typename dtype>
dtype get_one(){
  ABORT;
}


template<> inline
double get_zero<double>() { return 0.0; }

template<> inline
std::complex<double> get_zero< std::complex<double> >() { return std::complex<double>(0.0,0.0); }

template<> inline
double get_one<double>() { return 1.0; }

template<> inline
std::complex<double> get_one< std::complex<double> >() { return std::complex<double>(1.0,0.0); }

void read_matrix( const char *    filename,
                  int             myRank,
                  int             numPes, 
                  double * &      whole_A,
                  int &           m,
                  int &           b,
                  double * &      A, 
                  int &           mb);

void copy_lower(double const *  R_in,
                double *        R_out,
                int const       b,
                int const       r,
                int const       lda_in,
                int const       lda_out,
                int const       zero_square);

void copy_upper(double const *  R_in,
                double *        R_out,
                int const       b,
                int const       lda_in,
                int const       lda_out,
                int const       zero_square);

/**
 * \brief pack lower-triangular matrix from one rectangular buffer to 
 *        a triangular one
 *
 * \param[in] R_in buffer that contains initial lower-triangular matrix
 * \param[out] R_out buffer to which the lower-triangular matrix is copied
 * \param[in] b number of columns of R
 * \param[in] lda_in number of rows in the buffer R_in
 * \param[in] has_diag if 1 then diagonal explicitly stored
 **/
void pack_lower(double const *  R_in,
                double *        R_out,
                int const       b,
                int const       lda_in,
                int const       has_diag=0);

/**
 * \brief pack upper-triangular matrix from one rectangular buffer to 
 *        a triangular one
 *
 * \param[in] R_in buffer that contains initial upper-triangular matrix
 * \param[out] R_out buffer to which the upper-triangular matrix is copied
 * \param[in] b number of columns of R
 * \param[in] lda_in number of rows in the buffer R_in
 **/
void pack_upper(double const *  R_in,
                double *        R_out,
                int const       b,
                int const       lda_in);

/**
 * \brief unpack lower-triangular matrix from one rectangular buffer to 
 *        a triangular one
 *
 * \param[in] R_in buffer that contains initial lower-triangular matrix
 * \param[out] R_out buffer to which the lower-triangular matrix is copied
 * \param[in] b number of columns of R
 * \param[in] lda_out number of rows in the buffer R_out
 * \param[in] has_diag if 1 then diagonal explicitly stored
 **/
void unpack_lower(double const *  R_in,
                  double *        R_out,
                  int const       b,
                  int const       lda_out,
                  int const       has_diag=0);

/**
 * \brief unpack upper-triangular matrix from one rectangular buffer to 
 *        a triangular one
 *
 * \param[in] R_in buffer that contains initial upper-triangular matrix
 * \param[out] R_out buffer to which the upper-triangular matrix is copied
 * \param[in] b number of columns of R
 * \param[in] lda_out number of rows in the buffer R_out
 **/
void unpack_upper(double const *  R_in,
                  double *        R_out,
                  int const       b,
                  int const       lda_out);
/**
 * \brief sets a submatrix to zero
 *
 * \param[in] nrow number of rows to set to zero
 * \param[in] ncol number of columns to set to zero
 * \param[in] lda leading dimension length of matrix
 * \param[in] A matrix 
 */
void lda_zero(const int nrow,  const int ncol, 
              const int lda,   double * A);

/**
 * \brief reconstructs TAU values based on Householder vectors
 *
 * \param[in] strucuture: 'R' - rect, 'L' - lower-triangular, 
 *                        'U' - upper_triangular
 * \param[in] ncol number of Householder vectors
 * \param[in] nrow length of longest HH vector
 * \param[in] lda of matrix contaaining HH vectors
 * \param[in] Y matrix contaaining HH vectors
 * \param[out] tau nrow TAU values (preallocated)
 */
void tau_recon(const char     structure,
               const int      ncol,
               const int      nrow,
               const int      lda,
               double const * Y,
               double *       tau);

/**
 * \brief size of a packed upper-triangular matrix
 */
inline 
int psz_upr(int const n){
  return n*(n+1)/2;
}

/**
 * \brief size of a packed lower-triangular matrix
 */
inline 
int psz_lwr(int const n){
  return n*(n-1)/2;
}

void init_dist_sym_matrix(int n, 
                          int ipr,
                          int pr,
                          int ipc,
                          int pc,
                          int b,
                          double * full_A, 
                          double * loc_A); 
 
void print_dist_mat(int      nrow,
                    int      ncol, 
                    int      b,
                    int      ipr,
                    int      pr,
                    int      rpr,
                    int      ipc,
                    int      pc,
                    int      rpc,
                    MPI_Comm comm,
                    double * A,
                    int64_t  lda_A);

#endif
