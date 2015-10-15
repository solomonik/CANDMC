#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <string>
#include <ios>
#include <fstream>
#include <iostream>
#include <assert.h>

#include "../../alg/shared/util.h"
#include "../../alg/shared/comm.h"

using namespace std;

/**
 * \brief Benchmark lapack QR routines
 *
 * \param[in] m numer of rows in A
 * \param[in] k numer of columns in A
 * \param[in] b block size of A
 * \param[in] nprow numer of procesor rows
 * \param[in] npcol numer of procesor columns
 * \param[in] niter numer of iterations
 **/
void qr_seq_bench(int64_t const m,
                  int64_t const k, 
                  int64_t const b, 
                  int64_t const niter){
  printf("benchmarking sequential QR\n");
  double * A, * buf, * tau;
  double * B;
  double time_qr, time_ap, tick;
  int64_t i, iter;
  int info;

  int64_t seed_offset = 99900;

  srand48(seed_offset);

  assert(0==(posix_memalign((void**)&A,
          ALIGN_BYTES,
          m*k*sizeof(double))));
  assert(0==(posix_memalign((void**)&B,
          ALIGN_BYTES,
          m*k*sizeof(double))));  
  assert(0==(posix_memalign((void**)&buf,
          ALIGN_BYTES,
          m*k*sizeof(double))));
  assert(0==(posix_memalign((void**)&tau,
          ALIGN_BYTES,
          m*k*sizeof(double))));

  time_qr = 0.0;
  time_ap = 0.0;
  for (iter=0; iter<niter; iter++){
    for (i=0; i<m*k; i++){
      A[i] = drand48();
      B[i] = drand48();
    }
    tick = MPI_Wtime();
    cdgeqrf(m, k, A, m, tau, buf, m*k, &info);
    time_qr += MPI_Wtime() - tick;
    tick = MPI_Wtime();
    cdormqr('L', 'N', m, k, k, A, m, tau, B, m, buf, m*k, &info);
    time_ap += MPI_Wtime() - tick;
  }

  printf("DGEQRF on a " PRId64 "-by-" PRId64 " matrix took %lf seconds/iteration, at %lf GFlops\n",
          m, k, time_qr/niter, (2.*m*k*k-(2./3.)*k*k*k)/(time_qr/niter)*1.E-9);
  
  printf("DORMQR of " PRId64 "-by-" PRId64 " matrices took %lf seconds/iteration, at %lf GFlops\n",
          m, k, time_ap/niter, (2.*m*k*k-(2./3.)*k*k*k)/(time_ap/niter)*1.E-9);
  
  
  if (k*2 == m){
    time_qr = 0.0;
    time_ap = 0.0;
    for (iter=0; iter<niter; iter++){
      for (i=0; i<m*k; i++){
        A[i] = drand48();
        B[i] = drand48();
      }
      tick = MPI_Wtime();
      cdtpqrt(k, k, k, b, A, m, A+k, m, tau, b, buf, &info);
      time_qr += MPI_Wtime() - tick;
      tick = MPI_Wtime();
      cdtpmqrt('L', 'N', k, k, k, k, k, A, m, tau, k, 
               B, m, B+k, m, buf, &info);
      time_ap += MPI_Wtime() - tick;
    }

    printf("DTPQRT on a " PRId64 "-by-" PRId64 " matrix took %lf seconds/iteration, at %lf GFlops\n",
            m, k, time_qr/niter, (2.*m*k*k-(2./3.)*k*k*k)/(time_qr/niter)*1.E-9);
    printf("DTMPQRT on " PRId64 "-by-" PRId64 " matrices took %lf seconds/iteration, at %lf GFlops\n",
            m, k, time_ap/niter, (2.*m*k*k-(2./3.)*k*k*k)/(time_ap/niter)*1.E-9);

  }  

  free(A);
  free(B);
  free(buf);
  free(tau);
}

int main(int argc, char **argv) {
  int myRank, numPes;
  int64_t m, k, b, niter;

  CommData_t cdt_glb;
  INIT_COMM(numPes, myRank, 1, cdt_glb);

  if (argc == 2) {
    printf("Usage: ./exe <numer of rows> <numer of columns> ");
    printf("<block size> <numer of iterations>\n");
    ABORT;
  }
  if (argc > 1) m = atoi(argv[1]);
  else m = 64;
  if (argc > 2) k = atoi(argv[2]);
  else k = 32;
  if (argc > 3) b = atoi(argv[3]);
  else b = 8;
  if (argc > 4) niter = atoi(argv[4]);
  else niter = 10;
  if (myRank == 0){
    printf("m=" PRId64 ", k=" PRId64 ", b = " PRId64 ", niter = " PRId64 "\n",
            m,k,b,niter);
  }

  if (myRank == 0)
    qr_seq_bench(m, k, b, niter);

  COMM_EXIT;
  return 0;
}
