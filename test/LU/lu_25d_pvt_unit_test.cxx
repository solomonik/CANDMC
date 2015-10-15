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
void lu_25d_pvt_unit_test( int const        n,
                           int const        b_sm, 
                           int const        b_lrg, 
                           int const        myRank, 
                           int const        numPes, 
                           int const        c_rep,
                           CommData const   cdt){
  
  const CommData_t cdt_glb = cdt; 

  const int matrixDim = n;
  const int blockDim = b_sm;
  const int big_blockDim = b_lrg;

  const int seed = 9999;

  if (myRank == 0){ 
#ifdef PARTIAL_PVT
    printf("UNIT TESTING LU FACOTRIZATION OF SQUARE MATRIX WITH PARTIAL PIVOTING\n");
#else
    printf("UNIT TESTING LU FACOTRIZATION OF SQUARE MATRIX WITH TOURNAMENT PIVOTING\n");
#endif
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
    if (myRank == 0) printf("array_size mod block_size != 0!\n");
    ABORT;
  }

  const int num_blocks_dim = matrixDim/blockDim;
  const int num_pes_dim = sqrt(numPes/c_rep);

  if (myRank == 0){
    printf("NUM X BLOCKS IS %d\n", num_blocks_dim);
    //printf("NUM Y BLOCKS IS %d\n", num_blocks_dim);
    printf("NUM X PROCS IS %d\n", num_pes_dim);
    //printf("NUM Y PROCS IS %d\n", num_pes_dim);
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

  CommData_t cdt_kdir;
  CommData_t cdt_row;
  CommData_t cdt_col;
  CommData_t cdt_kcol;

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

  assert( posix_memalign((void**)&mat_A, 
          ALIGN_BYTES, 
          my_num_blocks_dim*my_num_blocks_dim*
          blockDim*blockDim*sizeof(double)) == 0);
  assert( posix_memalign((void**)&ans_LU, 
          ALIGN_BYTES, 
          my_num_blocks_dim*my_num_blocks_dim*
          blockDim*blockDim*sizeof(double)) == 0);
  assert( posix_memalign((void**)&buffer, 
          ALIGN_BYTES, 
          8*my_num_blocks_dim*my_num_blocks_dim*
          blockDim*blockDim*sizeof(double)) == 0);
  assert( posix_memalign((void**)&mat_P, 
          ALIGN_BYTES, 
          16*my_num_blocks_dim*blockDim*sizeof(int)) == 0);
  assert( posix_memalign((void**)&buffer_P, 
          ALIGN_BYTES, 
          16*my_num_blocks_dim*blockDim*sizeof(int)) == 0);

  if (layerRank == 0){
    for (ib=0; ib < my_num_blocks_dim; ib++){
      for (i=0; i < blockDim; i++){
        for (jb=0; jb < my_num_blocks_dim; jb++){
          for (j=0; j < blockDim; j++){
            srand48((num_pes_dim*ib + myCol)*matrixDim*blockDim 
                    + (num_pes_dim*jb + myRow)*blockDim + i*matrixDim+j
+seed);
                    mat_A[(ib*blockDim + i)*my_num_blocks_dim*blockDim + jb*blockDim+j] 
                                 = drand48();
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


  lu_25d_pvt_params_t p;

  p.pvt     = 1;
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
  p.cdt_kcol     = cdt_kcol;
#ifdef PARTIAL_PVT
  p.is_tnmt_pvt = 0;
#else
  p.is_tnmt_pvt = 1;
#endif

  lu_25d_pvt(&p, mat_A, mat_P, buffer_P, buffer);
  
  if (myRank == 0) printf("generating whole matrix for verification matrixDim=%d\n",matrixDim);
  double* whole_A = (double*)malloc(matrixDim*matrixDim*sizeof(double));
  double* computed_A = (double*)malloc(matrixDim*matrixDim*sizeof(double));
  double* computed_A2 = (double*)malloc(matrixDim*matrixDim*sizeof(double));
  int* pivot_A = (int*)malloc(matrixDim*sizeof(int));
  int* whole_P = (int*)malloc(matrixDim*sizeof(int));
  int* pivot_I = (int*)malloc(matrixDim*sizeof(int));
  for (i = 0; i < matrixDim; i++){
    pivot_I[i] = i;
    for (j = 0; j < matrixDim; j++){
      srand48(i*matrixDim +j + seed);
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
  pivot_conv(matrixDim, pivot_A, pivot_I);
#ifdef VERBOSE
  if (myRank==0){
    printf("solution is...\n");
    print_matrix(whole_A,matrixDim,matrixDim);
    for (i=0; i <matrixDim; i++){
      printf("P[%d] = %d\n", i, pivot_A[i]);
    }
  }
#endif
  #if 0
  int correct = 1;
    if (layerRank == 0){
      for (i = 0; i< blockDim*my_num_blocks_dim; i++){
        ib = i/blockDim;
        ib = ib*blockDim*num_pes_dim + myCol*blockDim + (i%blockDim);
        if (pivot_I[ib] != mat_P[i]){
          correct = 0;
          DEBUG_PRINTF("2.5D LU sent row %d to %d, BLAS sent it to %d\n",
                        ib,mat_P[i],pivot_I[ib]);
        }
      }
      if (correct){
        for (i = 0; i < blockDim*my_num_blocks_dim; i++){
          for (j = 0; j < blockDim*my_num_blocks_dim; j++){
            ib = i/blockDim;
            ib = ib*blockDim*num_pes_dim + myCol*blockDim + (i%blockDim);
            jb = j/blockDim;
            jb = jb*blockDim*num_pes_dim + myRow*blockDim + (j%blockDim);
            if (fabs(whole_A[ib*matrixDim + jb]
                     -mat_A[i*blockDim*my_num_blocks_dim+j]) > 0.0000001){
                //printf("[%d] Error at index row = %d, col = %d\n",myRank,jb,ib);
                DEBUG_PRINTF("[%d] LU[%d][%d] = %lf, should have been %lf\n",myRank,
                             jb,ib, 
                             mat_A[i*blockDim*my_num_blocks_dim+j],
                             whole_A[ib*matrixDim + jb]);
                correct = 0;
            }
          }
        } 
      } 
      else if (myRank == 0) printf("pivoting different from BLAS\n");
    }
    if (!correct && myRank == 0) printf("[%d] ERROR IN VERIFICATION\n", myRank);
    else {
      if (myRank == 0)
        printf("Verification of answer with respect to exact solution successful\n");
    }
  
  #endif
  if (layerRank == 0){
    std::fill(computed_A,computed_A+matrixDim*matrixDim,0.0);
    std::fill(whole_P,whole_P+matrixDim,0);
    for (i = 0; i < blockDim*my_num_blocks_dim; i++){
      ib = i/blockDim;
      ib = ib*blockDim*num_pes_dim + myCol*blockDim + (i%blockDim);
      for (j = 0; j < blockDim*my_num_blocks_dim; j++){
        jb = j/blockDim;
        jb = jb*blockDim*num_pes_dim + myRow*blockDim + (j%blockDim);
        computed_A[ib*matrixDim + jb] = mat_A[i*blockDim*my_num_blocks_dim+j];
        if (i == 0) whole_P[jb] = mat_P[j];
      }
    }
    MPI_Reduce(computed_A,computed_A2,matrixDim*matrixDim,MPI_DOUBLE,MPI_SUM,0,cdt_row.cm);
    if (myCol == 0) {
      MPI_Reduce(computed_A2,computed_A,matrixDim*matrixDim,MPI_DOUBLE,MPI_SUM,0,cdt_col.cm);
      MPI_Reduce(whole_P,pivot_A,matrixDim,MPI_INT,MPI_SUM,0,cdt_col.cm);
    }
    double * max_norm_tnmt = (double*)malloc(sizeof(double)); 
    double * frb_norm_tnmt = (double*)malloc(sizeof(double)); 
    if (myRank == 0){
#ifdef VERBOSE
      if (myRank==0){
        printf("lu_25d output is...\n");
        print_matrix(computed_A,matrixDim,matrixDim);
        for (i=0; i <matrixDim; i++){
          printf("P[%d] = %d\n", i, pivot_A[i]);
        }
      }
#endif
      backerr_lu(matrixDim,matrixDim,seed,computed_A,
                 pivot_A,frb_norm_tnmt,max_norm_tnmt,
                 0,0,matrixDim,matrixDim);
#ifndef PARTIAL_PVT
      printf("with tournmanet pivoting, backward norms |(A-LU)|, frobenius = %E, max = %E\n",
              frb_norm_tnmt[0],max_norm_tnmt[0]);
      if (frb_norm_tnmt[0] < 1.E-12 && max_norm_tnmt[0] <1.E-12){
        printf("test passed (parallel tournament pivoting test)\n");
      }
#else
      printf("with tournmanet pivoting, backward norms |(A-LU)|, frobenius = %E, max = %E\n",
              frb_norm_tnmt[0],max_norm_tnmt[0]);
      if (frb_norm_tnmt[0] < 1.E-12 && max_norm_tnmt[0] <1.E-12){
        printf("test passed (parallel partial pivoting test)\n");
      }
#endif
      //pvt_con_lu(matrixDim,matrixDim,seed,computed_A,pivot_A);
    }
  }


  /*FREE_CDT(cdt_kdir);
  FREE_CDT(cdt_row);
  FREE_CDT(cdt_col);*/
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
  if (sqrt(numPes)*sqrt(numPes) != numPes){
    if (myRank == 0)
      printf("LU test needs square processor grid... exiting.\n");
    return 0;
  }

#if 0 //SHARE_MIC > 1
  char sout[128];
  int offset = 0;
  offset += (60/SHARE_MIC)*(myRank>=numPes/2);
  sprintf(sout, "MIC_KMP_PLACE_THREADS=30c,4t,%dO",offset);
  putenv(sout);
#endif

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
  if (getCmdOption(argv, argv+argc, "-n")){
    n = atoi(getCmdOption(argv, argv+argc, "-n"));
    if (n <= b_lrg) n = b_lrg;
  } else n = MIN(128,b_lrg);
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
  lu_25d_pvt_unit_test(n, b_sm, b_lrg, myRank, numPes, c_rep, cdt_glb);
  
//  TAU_PROFILE_STOP(timer);
  MPI_Barrier(cdt_glb.cm);
  __CM(2, cdt_glb, numPes, num_iter, myRank);
  COMM_EXIT;
  return 0;
}

#endif
