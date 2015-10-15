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

#include "CANDMC.h"

using namespace std;

/**
 * \brief Test TSQR 
 *
 * \param[in] m number of rows in A
 * \param[in] b number of columns in A
 * \param[in] myRank rank in communicator column
 * \param[in] numPes number of processes in column
 * \param[in] req_id request id to use for send/recv
 * \param[in] comm MPI communicator for column
 **/
#ifdef READ_FILE
void qr_2d_unit_test( const char * filename, 
                      int64_t const myRank, 
                      int64_t const numPes, 
                      int64_t const req_id, 
                      CommData_t  cdt_row,
                      CommData_t  cdt_col,
                      CommData_t  cdt){
#else
void qr_2d_unit_test(int64_t const m,
                     int64_t const k, 
                     int64_t const b, 
                     int64_t const myRank, 
                     int64_t const numPes, 
                     int64_t const req_id, 
                     CommData_t  cdt_row,
                     CommData_t  cdt_col,
                     CommData_t  cdt){
#endif
  if (myRank == 0)  
    printf("unit testing parallel 2D butterfly QR...\n");
  double *A,*whole_A,*collect_YR,*whole_YR,*swork,*tau,*bw_A;
  double norm, yn2;
  int info;
  int64_t i,j,row,col,mb,kb,pass;
  int64_t npcol, nprow, mycol, myrow;
  npcol = cdt_row.np;
  nprow = cdt_col.np;
  mycol = cdt_row.rank;
  myrow = cdt_col.rank;


  int64_t seed_offset = 99900;

  mb = m / nprow;
  kb = k / npcol;
  assert(0==(posix_memalign((void**)&collect_YR,
          ALIGN_BYTES,
          2*m*k*sizeof(double))));
  assert(0==(posix_memalign((void**)&whole_YR,
          ALIGN_BYTES,
          2*m*k*sizeof(double))));
  assert(0==(posix_memalign((void**)&whole_A,
          ALIGN_BYTES,
          2*m*k*sizeof(double))));
  assert(0==(posix_memalign((void**)&A,
          ALIGN_BYTES,
          2*mb*kb*sizeof(double))));

  for (col=0; col<kb; col++){
    for (row=0; row<mb; row++){
      srand48(seed_offset +((myrow*b + (row%b) + (row/b)*b*nprow)
                          + (mycol*b + (col%b) + (col/b)*b*npcol)*m)*61);
      A[row+col*mb] = drand48()-.5;
      A[row+(kb+col)*mb] = A[row+col*mb];
    }
  }
  for (col=0; col<k; col++){
    for (row=0; row<m; row++){
      srand48(seed_offset + (row + col*m)*61);
      whole_A[row+col*m] = drand48()-.5;
      whole_A[row+(k+col)*m] = whole_A[row+col*m];
    }
  }

  assert(0==(posix_memalign((void**)&tau,
          ALIGN_BYTES,
          m*sizeof(double))));
  assert(0==(posix_memalign((void**)&swork,
          ALIGN_BYTES,
          m*k*sizeof(double))));
  assert(0==(posix_memalign((void**)&bw_A,
          ALIGN_BYTES,
          m*k*sizeof(double))));

  if (myRank == 0)
    printf("Performing 2D QR with butterfly updates on matrix [A,A]:\n");

  if (myRank == 0) printf("initial A:\n");
  double * A_ptr = A;
  /*for (int rr=0; rr<m/b; rr++){
    if (myRank == (rr%numPes)){
      print_matrix(A_ptr, b, 2*kb, mb);
      A_ptr+=b;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }*/
  //get YR via TSQR and reconstruction
  QR_butterfly_2D(A,mb,m,2*k,b,myRank,numPes,0,0,cdt_row,cdt_col,cdt,MIN(m,k));
  /* IMPORTANT: currently need to change sign of last R entry to align with LAPACK */
  /*A_ptr = A;
  if (myRank == 0) printf("post tree YR:\n");
  for (int rr=0; rr<m/b; rr++){
    if (myRank == (rr%numPes)){
      print_matrix(A_ptr, b, 2*kb, mb);
      A_ptr+=b;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }*/
  double fnorm = 0.0;
  for (col=0; col<kb; col++){
    for (row=0; row<mb; row++){
      if((myrow*b + (row%b) + (row/b)*b*nprow) > 
         (mycol*b + (col%b) + (col/b)*b*npcol)){
        fnorm += A[row+(kb+col)*mb]*A[row+(kb+col)*mb];
      } else {
        fnorm += MIN(pow(A[row+col*mb]-A[row+(kb+col)*mb],2),pow((A[row+col*mb]-A[row+(kb+col)*mb])/A[row+col*mb],2));
      }
    }
  }
  fnorm = sqrtf(fnorm);
  if (fnorm <1.E-6) pass = 1;
  else pass = 0;
  if (myRank == 0){
    printf("QR of [A,A] gives [Y/R_1,0/R_2], checking norm ||0/R_2-R_1||_F=%E\n",fnorm);
    if (pass) printf("Test passed.\n");
    else printf("Test FAILED!\n");
  }
  MPI_Barrier(cdt.cm);

  free(A);
  free(whole_A);
  free(whole_YR);
  free(collect_YR);
  free(tau);
  free(swork);
}

int main(int argc, char **argv) {
  int myRank, numPes;
  int64_t m, k, b, nprow, npcol;

  CommData_t cdt_glb;
  CommData_t cdt_row, cdt_col;
  INIT_COMM(numPes, myRank, 1, cdt_glb);

  /*string filename;
  if (argc != 2 ) 
    printf("Usage: mpirun -np <num procs> ./exe <matrix file>\n");

  if (argc == 2) {
    filename.append(argv[1]);
  }
  qr_2d_unit_test(filename.c_str(), myRank, numPes, 0, cdt_glb);
  */

  nprow = sqrt(numPes);
  while (numPes % nprow != 0) nprow++;
  npcol = numPes/nprow;
  if (argc == 1) {
    b = 2;
    m = 8*nprow;
    k = 4*npcol;
  } else if (argc > 3) {
    m = atoi(argv[1]);
    k = atoi(argv[2]);
    b = atoi(argv[3]);
    if (argc > 4)  nprow = atoi(argv[4]);
    assert(m > 0);
    assert(b > 0);
    assert(k > 0);
    assert(nprow > 0);
    assert(nprow <= numPes);
  } else {
    printf("Usage: mpirun -np <num procs> ./exe <number of rows>");
    printf(" <number of columns> <block size> <nprow>\n");
    ABORT;
  }
  npcol = numPes/nprow;
  if (myRank == 0){
    printf("m=" PRId64 ", k=" PRId64 ", b = " PRId64 ", nprow = " PRId64 ", npcol = " PRId64 "\n",
            m,k,b,nprow,npcol);
  }
  SETUP_SUB_COMM(cdt_glb, (cdt_row), 
                 myRank/nprow, 
                 myRank%nprow, 
                 npcol);
  SETUP_SUB_COMM(cdt_glb, (cdt_col), 
                 myRank%nprow, 
                 myRank/nprow, 
                 nprow);
  
  qr_2d_unit_test(m, k, b, myRank, numPes, 0, cdt_row, cdt_col, cdt_glb);


  COMM_EXIT;
  return 0;
}
