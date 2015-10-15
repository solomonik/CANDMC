//#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include "tnmt_pvt.h"
#include "lu_25d_pvt.h"
#include "unit_test.h"
#include "../shared/util.h"

#define MAX_OFF_DIAG	1.1

/* test parallel tournament pivoting
 * n is the test matrix dimension 
 * b_sm is the small block dimension 
 * b_lrg is the large block dimension */
void pvt_unit_test(int const 		n,
		   int const		b_sm, 
		   int const		b_lrg, 
		   int const		myRank, 
		   int const		numPes, 
		   int const		c_rep,
		   CommData const 	*cdt){
  
  const CommData_t *cdt_glb = cdt; 

  int info;
  const int matrixDim = n;
  const int blockDim = b_sm;
  const int big_blockDim = b_lrg;

  const int seed = 9999;

  if (myRank == 0){ 
    printf("UNIT TESTING LU FACOTRIZATION OF SQUARE MATRIX WITH PIVOTING\n");
    printf("MATRIX DIMENSION IS %d\n", matrixDim);
    printf("BLOCK DIMENSION IS %d\n", blockDim);
    printf("BIG BLOCK DIMENSION IS %d\n", big_blockDim);
    printf("REPLICATION FACTOR, C, IS %d\n", c_rep);
#ifdef USE_DCMF
    printf("USING DCMF FOR COMMUNICATION\n");
#else
    printf("USING MPI FOR COMMUNICATION\n");
#endif
  }

  if (matrixDim < blockDim || matrixDim % blockDim != 0) {
    if (myRank == 0) printf("array_size \% block_size != 0!\n");
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
      printf("NUMBER OF BLOCKS MUST BE DIVISBLE BY THE NUMBER OF PROCESSORS\n");
    ABORT;
  }
 
  int layerRank, intraLayerRank, myRow, myCol;

  CommData_t *cdt_kdir = (CommData_t*)malloc(sizeof(CommData_t));  
  CommData_t *cdt_row = (CommData_t*)malloc(sizeof(CommData_t));  
  CommData_t *cdt_col = (CommData_t*)malloc(sizeof(CommData_t));  

  RSETUP_KDIR_COMM(myRank, numPes, c_rep, cdt_kdir, layerRank, intraLayerRank, 2, 1);
  RANK_PRINTF(0,myRank," set up kdir comms\n");
  RSETUP_LAYER_COMM(num_pes_dim, layerRank, intraLayerRank, cdt_row, cdt_col, myRow, myCol, 2, 0);
  RANK_PRINTF(0,myRank," set up layer comms\n");
  const int my_num_blocks_dim = num_blocks_dim/num_pes_dim;
  const int num_big_blocks_dim = matrixDim/big_blockDim;
  const int num_small_in_big_blk = big_blockDim/blockDim;
  const int my_num_small_in_big_blk = num_small_in_big_blk/num_pes_dim;
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
			8*my_num_blocks_dim*my_num_blocks_dim*
			blockDim*blockDim*sizeof(double)) == 0);
  assert(posix_memalign((void**)&mat_P, 
			ALIGN_BYTES, 
			16*my_num_blocks_dim*blockDim*sizeof(int)) == 0);
  assert(posix_memalign((void**)&buffer_P, 
			ALIGN_BYTES, 
			16*my_num_blocks_dim*blockDim*sizeof(int)) == 0);

  if (layerRank == 0){
    for (ib=0; ib < my_num_blocks_dim; ib++){
      for (i=0; i < blockDim; i++){
	for (jb=0; jb < my_num_blocks_dim; jb++){
	  for (j=0; j < blockDim; j++){
	    srand48((num_pes_dim*ib + myCol)*matrixDim*blockDim 
		  + (num_pes_dim*jb + myRow)*blockDim + i*matrixDim+j +seed);
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

  GLOBAL_BARRIER(cdt_glb);


  lu_25d_pvt_params_t p;

  p.pvt			= 1;
  p.myRank 		= myRank;
  p.c_rep 		= c_rep;
  p.matrixDim 		= matrixDim;
  p.blockDim 		= blockDim;
  p.big_blockDim 	= big_blockDim;
  p.num_pes_dim 	= num_pes_dim;
  p.layerRank 		= layerRank;
  p.myRow 		= myRow;
  p.myCol 		= myCol;
  p.cdt_row 		= cdt_row;
  p.cdt_col 		= cdt_col;
  p.cdt_kdir 		= cdt_kdir;

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
  cdgetrf(matrixDim, matrixDim, whole_A, matrixDim, pivot_A, &info);
  pivot_conv(matrixDim, pivot_A, pivot_I);
#ifdef VERBOSE
  if (myRank==0){
    printf("solution is...\n");
    print_matrix(whole_A,matrixDim,matrixDim);
  }
#endif
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
  correct = 1;
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
    REDUCE(computed_A,computed_A2,matrixDim*matrixDim,COMM_DOUBLE_T,COMM_OP_SUM,0,cdt_row);
    if (myCol == 0) {
      REDUCE(computed_A2,computed_A,matrixDim*matrixDim,COMM_DOUBLE_T,COMM_OP_SUM,0,cdt_col);
      REDUCE(whole_P,pivot_A,matrixDim,COMM_INT_T,COMM_OP_SUM,0,cdt_col);
    }
    double * frb_norm, *max_norm;
    double * max_norm_tnmt = (double*)malloc(sizeof(double)); 
    double * frb_norm_tnmt = (double*)malloc(sizeof(double)); 
    if (myRank == 0){
#ifdef VERBOSE
      print_matrix(computed_A,matrixDim,matrixDim);
      for (i=0; i <matrixDim; i++){
	printf("P[%d] = %d\n", i, pivot_A[i]);
      }
#endif
      backerr_lu(matrixDim,matrixDim,seed,computed_A,
		 pivot_A,frb_norm_tnmt,max_norm_tnmt,
		 0,0,matrixDim,matrixDim);
      printf("with tournmanet pivoting, backward norms |(A-LU)|, frobenius = %E, max = %E\n",
		frb_norm_tnmt[0],max_norm_tnmt[0]);
      if (frb_norm_tnmt[0] < 1.E-12 && max_norm_tnmt[0] <1.E-12){
	printf("test passed (parallel tournament pivoting test)\n");
      }
      pvt_con_lu(matrixDim,matrixDim,seed,computed_A,pivot_A);
    }
  }


  /*FREE_CDT(cdt_kdir);
  FREE_CDT(cdt_row);
  FREE_CDT(cdt_col);*/
} /* end function main */


