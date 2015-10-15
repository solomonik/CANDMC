/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <math.h>
#include "CANDMC.h"
#include "unit_test.h"

#define MAX_OFF_DIAG  1.1

void seq_square_lu(double *A,
        int *pivot_mat,
        const int dim){
  int info;
  cdgetrf(dim, dim, A, dim, pivot_mat, &info);
}

/* test parallel tournament pivoting
 * n is the test matrix dimension 
 * b_sm is the small block dimension 
 * b_lrg is the large block dimension */
void lu_25d_unit_test(int const     n,
                      int const     b_sm, 
                      int const     b_lrg, 
                      int const     myRank, 
                      int const     numPes, 
                      int const     c_rep,
                      CommData const cdt){
  
  const CommData_t cdt_glb = cdt; 

  const int matrixDim = n;
  const int blockDim = b_sm;
  const int big_blockDim = b_lrg;

  if (myRank == 0){ 
    printf("UNIT TESTING LU FACOTRIZATION OF SQUARE MATRIX WITH NO PIVOTING\n");
    printf("MATRIX DIMENSION IS %d, ", matrixDim);
    printf("BLOCK DIMENSION IS %d, ", blockDim);
    printf("BIG BLOCK DIMENSION IS %d\n", big_blockDim);
    printf("REPLICATION FACTOR, C, IS %d\n", c_rep);
#ifdef USE_DCMF
    printf("USING DCMF FOR COMMUNICATION\n");
#else
    printf("USING MPI FOR COMMUNICATION\n");
#endif
  }

  if (matrixDim < blockDim || matrixDim % blockDim != 0) {
    if (myRank == 0) printf("array_size block_size != 0!\n");
    ABORT;
  }

  const int num_blocks_dim = matrixDim/blockDim;
  const int num_pes_dim = sqrt(numPes/c_rep);

  if (myRank == 0){
    printf("NUM X BLOCKS IS %d\n", num_blocks_dim);
    printf("NUM Y BLOCKS IS %d\n", num_blocks_dim);
    printf("NUM X PROCS IS %d\n", num_pes_dim);
    printf("NUM Y PROCS IS %d\n", num_pes_dim);
  }

  if (num_pes_dim*num_pes_dim != numPes/c_rep || numPes%c_rep > 0 || c_rep > num_pes_dim){
    if (myRank == 0) 
      printf("ERROR: PROCESSOR GRID MISMATCH\n");
    ABORT;
  }
  if (big_blockDim%(num_pes_dim*blockDim)!=0  || big_blockDim<=0 || big_blockDim>matrixDim){
    if (myRank == 0) 
      printf("ERROR: BIG BLOCK DIMENSION MUST BE num_pes_dim*blockDim*x for some x.\n");
    ABORT;
  }

  if (num_blocks_dim%num_pes_dim != 0){
    if (myRank == 0) 
      printf("NUMBER OF BLOCKS MUST BE DIVISBLE BY THE 2D PROCESSOR GRID DIMENSION\n");
    ABORT;
  }
 
  int layerRank, intraLayerRank, myRow, myCol;

  CommData_t cdt_kdir;// = (CommData_t*)malloc(sizeof(CommData_t));  
  CommData_t cdt_row;// = (CommData_t*)malloc(sizeof(CommData_t));  
  CommData_t cdt_col;// = (CommData_t*)malloc(sizeof(CommData_t));  
  CommData_t cdt_kcol;// = (CommData_t*)malloc(sizeof(CommData_t));  

  RSETUP_KDIR_COMM(myRank, numPes, c_rep, cdt_kdir, layerRank, intraLayerRank);
  RANK_PRINTF(0,myRank," set up kdir comms\n");
  RSETUP_LAYER_COMM(num_pes_dim, layerRank, intraLayerRank, cdt_row, cdt_col, myRow, myCol);
  RANK_PRINTF(0,myRank," set up layer comms\n");
  SETUP_SUB_COMM(cdt_glb, cdt_kcol, (layerRank*num_pes_dim+myRow), 
                 myCol, (c_rep*num_pes_dim));
  const int my_num_blocks_dim = num_blocks_dim/num_pes_dim;
  int i,j,ib,jb;

  double * mat_A, * ans_LU, * buffer;
  int * mat_P, * buffer_P;

  assert(posix_memalign((void**)&mat_A, 
      ALIGN_BYTES, 
      my_num_blocks_dim*my_num_blocks_dim*
      blockDim*blockDim*sizeof(double)) == 0);
  assert(posix_memalign((void**)&ans_LU, 
      ALIGN_BYTES, 
      my_num_blocks_dim*my_num_blocks_dim*
      blockDim*blockDim*sizeof(double)) == 0);
  assert(posix_memalign((void**)&buffer, 
      ALIGN_BYTES, 
      160*my_num_blocks_dim*my_num_blocks_dim*
      blockDim*blockDim*sizeof(double)) == 0);
  assert(posix_memalign((void**)&mat_P, 
      ALIGN_BYTES, 
      160*my_num_blocks_dim*blockDim*sizeof(int)) == 0);
  assert(posix_memalign((void**)&buffer_P, 
      ALIGN_BYTES, 
      160*my_num_blocks_dim*blockDim*sizeof(int)) == 0);

  if (myRank == 0) printf("TESTING NO PIVOTING LU (DIAGONALLY DOMINANT MATRIX)\n");

  if (layerRank == 0){
    for (ib=0; ib < my_num_blocks_dim; ib++){
      for (i=0; i < blockDim; i++){
  for (jb=0; jb < my_num_blocks_dim; jb++){
    for (j=0; j < blockDim; j++){
      srand48((num_pes_dim*ib + myCol)*matrixDim*blockDim 
      + (num_pes_dim*jb + myRow)*blockDim + i*matrixDim+j);
      if (ib == jb && i == j && myRow == myCol)
        mat_A[(ib*blockDim + i)*my_num_blocks_dim*blockDim + jb*blockDim+j] 
      = (MAX_OFF_DIAG+drand48())*matrixDim;
      else
        mat_A[(ib*blockDim + i)*my_num_blocks_dim*blockDim + jb*blockDim+j] 
                     = drand48();
      /*srand48((num_pes_dim*ib + myCol)*matrixDim*blockDim 
      + (num_pes_dim*jb + myRow)*blockDim + i*matrixDim+j);
      mat_A[(ib*blockDim + i)*my_num_blocks_dim*blockDim + jb*blockDim+j] = drand48();*/
    }
  }
      }
    }
  }
  else { /* If I am not the first layer I will be receiving whatever data I need from
    * layer 0, and my updates will just be accumulating to the Schur complement */
    for (i=0; i < my_num_blocks_dim*my_num_blocks_dim*blockDim*blockDim; i++)
      mat_A[i] = 0.0;
  }

  MPI_Barrier(cdt_glb.cm);

/*#ifdef TAU
  TAU_PROFILE_TIMER(timer, "main", "int (int, char**)", TAU_USER);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(myRank);
  TAU_PROFILE_SET_CONTEXT(0);
#endif

  INIT_COMM_TIME//;
  INIT_IDLE_TIME//;
  
  MPI_Barrier(cdt_glb.cm);
*/

  lu_25d_pvt_params_t p;

  p.pvt     = 0;
  p.myRank    = myRank;
  p.c_rep     = c_rep;
  p.matrixDim     = matrixDim;
  p.blockDim    = blockDim;
  p.big_blockDim  = big_blockDim;
  p.num_pes_dim   = num_pes_dim;
  p.layerRank     = layerRank;
  p.myRow     = myRow;
  p.myCol     = myCol;
  p.cdt_row     = cdt_row;
  p.cdt_col     = cdt_col;
  p.cdt_kdir    = cdt_kdir;
  p.cdt_kcol    = cdt_kcol;
  p.is_tnmt_pvt = 1;

  lu_25d_pvt(&p, mat_A, mat_P, buffer_P, buffer);
  
  if (myRank == 0) printf("generating whole matrix for verification matrixDim=%d\n",matrixDim);
  double* whole_A = (double*)malloc(matrixDim*matrixDim*sizeof(double));
  int* pivot_A = (int*)malloc(matrixDim*sizeof(int));
  for (i = 0; i < matrixDim; i++){
    for (j = 0; j < matrixDim; j++){
      srand48(i*matrixDim +j);
      if (i==j)
  whole_A[i*matrixDim+j] = (MAX_OFF_DIAG+drand48())*matrixDim;
      else
  whole_A[i*matrixDim+j] = drand48();
    }
  }

#ifdef VERBOSE
  if (myRank==0){
    printf("matrix is...\n");
    print_matrix(whole_A,matrixDim,matrixDim);
  }
#endif
  /* solve the entire problem sequentially */
  seq_square_lu(whole_A,pivot_A,matrixDim); 
#ifdef VERBOSE
  if (myRank==0){
    printf("solution is...\n");
    print_matrix(whole_A,matrixDim,matrixDim);
  }
#endif
  int correct = 1;
  if (layerRank == 0){
    for (i = 0; i < blockDim*my_num_blocks_dim; i++){
      for (j = 0; j < blockDim*my_num_blocks_dim; j++){
  ib = i/blockDim;
  ib = ib*blockDim*num_pes_dim + myCol*blockDim + (i%blockDim);
  jb = j/blockDim;
  jb = jb*blockDim*num_pes_dim + myRow*blockDim + (j%blockDim);
  if (fabs(whole_A[ib*matrixDim + jb]
     -mat_A[i*blockDim*my_num_blocks_dim+j]) > 0.0000001){
      printf("[%d] Error at index row = %d, col = %d\n",myRank,jb,ib);
      printf("[%d] LU[%d][%d] = %lf, should have been %lf\n",myRank,
        jb,ib, 
        mat_A[i*blockDim*my_num_blocks_dim+j],
        whole_A[ib*matrixDim + jb]);
      correct = 0;
  }
      }
    }
  }
  if (!correct) printf("[%d] ERROR IN VERIFICATION\n", myRank);
  else {
    if (myRank == 0)
      printf("Verification of answer successful\n");
  }
  FREE_CDT((&cdt_kdir));
  FREE_CDT((&cdt_row));
  FREE_CDT((&cdt_col));
} /* end function main */


#ifndef UNIT_TEST

static
char* getCmdOption(char ** begin, char ** end, const std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

int main(int argc, char **argv) {
  int myRank, numPes, c_rep, num_iter, pvt, nwarm;
  int64_t b_sm, b_lrg, n;

  CommData_t cdt_glb;
  RINIT_COMM(numPes, myRank, 4, 4, cdt_glb);
  

  if (getCmdOption(argv, argv+argc, "-n")){
    n = atoi(getCmdOption(argv, argv+argc, "-n"));
    if (n <= 0) n = 128;
  } else n = 128;

  if (getCmdOption(argv, argv+argc, "-num_iter")){
    num_iter = atoi(getCmdOption(argv, argv+argc, "-num_iter"));
    if (num_iter <= 0) num_iter = 1;
  } else num_iter = 3;

  if (getCmdOption(argv, argv+argc, "-nwarm")){
    nwarm = atoi(getCmdOption(argv, argv+argc, "-nwarm"));
    if (nwarm <= 0) nwarm = 0;
  } else nwarm = 1;

  if (getCmdOption(argv, argv+argc, "-c_rep")){
    c_rep = atoi(getCmdOption(argv, argv+argc, "-c_rep"));
    if (c_rep <= 1) c_rep = 1;
  } else {
    c_rep = 1;
    if (sqrt(numPes/c_rep)*sqrt(numPes/c_rep) != numPes/c_rep){
      if (numPes >= 8 && numPes%2 == 0) c_rep = 2;
    }
  }
  if (getCmdOption(argv, argv+argc, "-b_sm")){
    b_sm = atoi(getCmdOption(argv, argv+argc, "-b_sm"));
    if (b_sm < 1) b_sm = 4;
  } else b_sm = 4;
  if (getCmdOption(argv, argv+argc, "-b_lrg")){
    b_lrg = atoi(getCmdOption(argv, argv+argc, "-b_lrg"));
    if (b_lrg < b_sm*sqrt(numPes/c_rep)) 
      b_lrg = b_sm*sqrt(numPes/c_rep);
  } else 
      b_lrg = b_sm*sqrt(numPes/c_rep);
  if (getCmdOption(argv, argv+argc, "-pvt")){
    pvt = atoi(getCmdOption(argv, argv+argc, "-pvt"));
    if (pvt < 1) pvt = 3;
  } else pvt = 3;
/*#ifdef TAU
  TAU_PROFILE_TIMER(timer, "main", "int (int, char**)", TAU_USER);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(myRank);
  TAU_PROFILE_SET_CONTEXT(0);
#endif
  */
  if (sqrt(numPes/c_rep)*sqrt(numPes/c_rep) != numPes/c_rep){
    if (myRank == 0)
      printf("LU test needs square processor grid... exiting.\n");
    return 0;
  }


  MPI_Barrier(cdt_glb.cm);
  lu_25d_unit_test(n, b_sm, b_lrg, myRank, numPes, c_rep, cdt_glb);
  
//  TAU_PROFILE_STOP(timer);
  MPI_Barrier(cdt_glb.cm);
  __CM(2, cdt_glb, numPes, num_iter, myRank);
  COMM_EXIT;
  return 0;
}

#endif
