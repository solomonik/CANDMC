#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <assert.h>
#include <string>
#include <fstream>
#include "CANDMC.h"

/*
#ifndef __C_SRC
#define __C_SRC
#endif
*/
#define NUM_ITER 5

//proper modulus for 'a' in the range of [-b inf]
#define WRAP(a,b)       ((a + b)%b)
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )

int main(int argc, char **argv) {
/*void pbm() {

  int argc;
  char **argv;*/
  int myRank, numPes;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Request req[4];
  MPI_Status status[4];

  if (argc < 5 || argc > 6) {
    if (myRank == 0) 
      printf("%s [mat_dim] [pe_mat_lda] [blk_dim] [number of iterations] [file string name]\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int matrixDim = atoi(argv[1]);
  int blockDim = atoi(argv[2]);		//  matrix dimension of local matrix?
  int sbDim = atoi(argv[3]);			// block dimension (algorithmic, tunable parameter)?
  int num_iter = atoi(argv[4]);

  if (myRank == 0){ 
    printf("PDPOTRF OF SQUARE MATRIX\n");
    printf("MATRIX DIMENSION IS %d\n", matrixDim);
    printf("numProcessors - %d\n", numPes);
    printf("BLOCK DIMENSION IS %d\n", sbDim);
    printf("PERFORMING %d ITERATIONS\n", num_iter);
  }

  if ((matrixDim < blockDim) || (matrixDim % blockDim != 0)) {
    if (myRank == 0) printf("array_size_X BAD block_size_X != 0!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int num_blocks_dim = matrixDim / blockDim;		// Number of local blocks along each dimension, another way of figuring out my processor grid dimension, right?

  if (myRank == 0){
    printf("NUM X BLOCKS IS %d\n", num_blocks_dim);
    printf("NUM Y BLOCKS IS %d\n", num_blocks_dim);
  }

  if (num_blocks_dim*num_blocks_dim != numPes){
    if (myRank == 0) printf("NUMBER OF BLOCKS MUST BE EQUAL TO NUMBER OF PROCESSORS\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int  myRow = myRank / num_blocks_dim;
  int  myCol = myRank % num_blocks_dim;
  int iter, i, j, k, l, blk_row, blk_col, blk_row_offset, write_offset, offset;

  double * mat_A = (double*)malloc(blockDim*blockDim*sizeof(double));
  
  double * temp;
  double ans_verify;


  int icontxt, info;
  int iam, inprocs;
  Cblacs_pinfo(&iam,&inprocs);
  Cblacs_get(-1, 0, &icontxt);
  Cblacs_gridinit(&icontxt, "Row", num_blocks_dim, num_blocks_dim);

  int desc_a[9];
  cdescinit(desc_a, matrixDim, matrixDim,
                    sbDim, sbDim,
                    0,  0,
                    icontxt, blockDim, 
                                 &info);
  assert(info==0);

  std::string fptrStrTotal = argv[5];
  fptrStrTotal += ".txt";
  std::ofstream fptrTotal;
  if (myRank == 0)
  {
    fptrTotal.open(fptrStrTotal.c_str());
  }

  double startTime, iterTimeLocal, iterTimeGlobal, totalTime;
  totalTime = 0.0;
  for (iter=0; iter < num_iter; iter++){
    srand48(1234*myRank);
    for (i=0; i < blockDim; i++){
      for (j=0; j < blockDim; j++){
        mat_A[i*blockDim+j] = (((i==j) && (myRow == myCol)) ? drand48() + matrixDim*4 : drand48());
      }
    }
    startTime = MPI_Wtime();
    cpdpotrf('U', matrixDim, mat_A, 1, 1, desc_a, &info);
    iterTimeLocal=MPI_Wtime() - startTime;
    if (info != 0)
    {
      if (info < 0) {printf("Bad below!\n");}
      if (info > 0) {printf("Bad above!\n");}
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
    MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myRank == 0)
    {
      printf("numPes - %d, iter - %d, matrixDim - %d, time - %g\n", numPes, iter, matrixDim, iterTimeGlobal);
      fptrTotal << numPes << "\t" << iter << "\t" << matrixDim << "\t" << iterTimeGlobal << std::endl;
    }
    totalTime += iterTimeGlobal;
  }

  if (myRank == 0)
  {
    fptrTotal.close();
  }

  if(myRank == 0) {
    printf("Completed %u iterations\n", iter);
    printf("Time elapsed per iteration: %f\n", totalTime/num_iter);
    printf("Gigaflops: %f\n", ((2./3.)*matrixDim*matrixDim*matrixDim)/
                                (totalTime/num_iter)*1E-9);
  }

  MPI_Finalize();
  return 0;
} /* end function main */


