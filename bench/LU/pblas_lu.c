#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <assert.h>

#ifndef __C_SRC
#define __C_SRC
#endif

#define NUM_ITER 5

//proper modulus for 'a' in the range of [-b inf]
#define WRAP(a,b)       ((a + b)%b)
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )

void Cblacs_pinfo(int*,int*);

void Cblacs_get(int,int,int*);

int Cblacs_gridinit(int*,char*,int,int);

void descinit(int *, const int *,
                const int *, const int *,
                const int *, const int *,
                const int *, const int *,
                const int *, int *);

static void cdesc_init(int * desc, 
                      const int m,      const int n,
                      const int mb,     const int nb,
                      const int irsrc,  const int icsrc,
                      const int ictxt,  const int LLD,
                                        int * info){
  descinit(desc,&m,&n,&mb,&nb,&irsrc,&icsrc,
             &ictxt, &LLD, info);
}

static void  pdgetrf(int *,     int *,
                     double *,  int *,
                     int *,     int *,
                     int *,     int *);

static void cpdgetrf(int m,     int n,
                     double *A, int ia,
                     int ja,    int * desca,
                     int *IPIV, int * info){
  pdgetrf(&m,&n,A,&ia,&ja,desca,IPIV,info);
}


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

  const  int log_numPes = uint_log2(numPes);


  if (argc < 4 || argc > 5) {
    if (myRank == 0) 
      printf("%s [log2_mat_dim] [log2_pe_mat_lda] [log2_blk_dim] [number of iterations]\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  const  int log_matrixDim = atoi(argv[1]);
  const  int log_blockDim = atoi(argv[2]);
  const  int log_sbDim = atoi(argv[3]);
  const  int matrixDim = 1<<log_matrixDim;
  const  int blockDim = 1<<log_blockDim;
  const  int sbDim = 1<<log_sbDim;

  int num_iter;
  if (argc > 4) num_iter = atoi(argv[4]);
  else num_iter = NUM_ITER;

  if (myRank == 0){ 
    printf("PDGETRFOF SQUARE MATRIX\n");
    printf("MATRIX DIMENSION IS %d\n", matrixDim);
    printf("BLOCK DIMENSION IS %d\n", sbDim);
    printf("PERFORMING %d ITERATIONS\n", num_iter);
#ifdef RAND
    printf("WITH RANDOM DATA\n");
#else
    printf("WITH DATA=INDEX\n");
#endif
  }

  if (matrixDim < blockDim || matrixDim % blockDim != 0) {
    if (myRank == 0) printf("array_size_X \% block_size_X != 0!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  if (matrixDim < blockDim || matrixDim % blockDim != 0) {
    if (myRank == 0) printf("array_size_Y \% block_size_Y != 0!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  const  int log_num_blocks_dim = log_matrixDim - log_blockDim;
  const  int num_blocks_dim = 1<<log_num_blocks_dim;

  if (myRank == 0){
    printf("NUM X BLOCKS IS %d\n", num_blocks_dim);
    printf("NUM Y BLOCKS IS %d\n", num_blocks_dim);
  }

  if (num_blocks_dim*num_blocks_dim != numPes){
    if (myRank == 0) printf("NUMBER OF BLOCKS MUST BE EQUAL TO NUMBER OF PROCESSORS\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

   int const myRow = myRank / num_blocks_dim;
   int const myCol = myRank % num_blocks_dim;
   int iter, i, j, k, l, blk_row, blk_col, blk_row_offset, write_offset, offset;

  double * mat_A = (double*)malloc(blockDim*blockDim*sizeof(double));
  int * mat_P = (int*)malloc(2*matrixDim*sizeof(int));
  
  double * temp;
  double ans_verify;


  int icontxt, info;
  int iam, inprocs;
  Cblacs_pinfo(&iam,&inprocs);
  Cblacs_get(-1, 0, &icontxt);
  Cblacs_gridinit(&icontxt, "Row", num_blocks_dim, num_blocks_dim);

  int desc_a[9];
  cdesc_init(desc_a, matrixDim, matrixDim,
                    sbDim, sbDim,
                    0,  0,
                    icontxt, blockDim, 
                                 &info);
  assert(info==0);

                                    
  double startTime, endTime, totalTime;
  totalTime = 0.0;
  for (iter=0; iter < num_iter; iter++){
    srand48(1234);
    for (i=0; i < blockDim; i++){
      for (j=0; j < blockDim; j++){
        mat_A[i*blockDim+j] = drand48();
      }
    }
    startTime = MPI_Wtime();
    cpdgetrf(matrixDim, matrixDim, mat_A, 1, 1, desc_a, mat_P, &info);
    endTime = MPI_Wtime();
    totalTime += endTime -startTime;
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


