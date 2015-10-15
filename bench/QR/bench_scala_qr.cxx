#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <assert.h>
#include "CANDMC.h"


//proper modulus for 'a' in the range of [-b inf]
#define WRAP(a,b)	((a + b)%b)
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )
#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )


template <typename dtype>
inline void init_matrices(int const m, 
                   int const n,
                   int const ipr,
                   int const pr,
                   int const ipc,
                   int const pc,
                   int const b,
                   dtype * full_A, 
                   dtype * loc_A){
  int loc_off, i ,j;
  loc_off = 0;
  for (i=0; i<n; i++){
    for (j=0; j<m; j++){
      full_A[i*m+j] = drand48();
      if ((i/b)%pc == ipc && (j/b)%pr == ipr){
        loc_A[loc_off] = full_A[i*m+j];
        loc_off++;
      }
    }
  }
}

template <typename dtype>
inline void bench_scala_qr(int64_t m, int64_t n, int64_t b, int64_t niter, int64_t verify, int64_t pr, int tdorgqr){
dtype * loc_A, * full_A, * work;
  int myRank, numPes;
  int64_t pc, ipr, ipc, i, j, loc_off, nz;
  dtype * TAU2, * loc_TAU2;
  int64_t * iwork, * iclustr, * ifail;
  int icontxt, iam, inprocs;
  int64_t iter, lwork;
  int info;
  char cC = 'C';
  int desc_A[9], desc_EC[9];
  double time;


  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  pc = numPes / pr;
  ipc = myRank / pr;
  ipr = myRank % pr;
 
  loc_A    = (dtype*)malloc(m*n*sizeof(dtype)/numPes);
  if (verify){
    full_A  = (dtype*)malloc(m*n*sizeof(dtype));
    TAU2  = (dtype*)malloc(n*sizeof(dtype));
  }
  loc_TAU2  = (dtype*)malloc(n*sizeof(dtype)/pc);
  lwork       = MAX(5*m*n/numPes,300*(n+m));
  work        = (dtype*)malloc(lwork*sizeof(dtype));
  iwork       = (int64_t*)malloc(300*(m+n)*sizeof(int64_t));

  srand48(666);

  Cblacs_pinfo(&iam,&inprocs);
  Cblacs_get(-1, 0, &icontxt);
  Cblacs_gridinit(&icontxt, &cC, pr, pc);
  cdescinit(desc_A, m, n,
		        b, b,
		        0, 0,
		        icontxt, m/pr, 
				    &info);
  assert(info==0);

  if (verify && m*n*n < 1E9){
     if (myRank == 0) printf("Verifying cpgeqrf correctness\n");
    init_matrices(m,n,ipr,pr,ipc,pc,b,full_A,loc_A); 
    cxgeqrf<dtype>(m,n,full_A,m,TAU2,work,lwork,&info);
    cpxgeqrf<dtype>(m,n,loc_A,1,1,desc_A,loc_TAU2,work,lwork,&info);
    loc_off = 0;
    for (i=0; i<n; i++){
      for (j=0; j<m; j++){
        if ((i/b)%pc == ipc && (j/b)%pr == ipr){
          dtype diff = full_A[i*m+j]-loc_A[loc_off];
          if (GET_REAL(diff)/GET_REAL(full_A[i*m+j]) > 1.E-6)
            printf("incorrect answer " PRId64 " scalapack computed %E, lapack computed %E\n",
                    i, GET_REAL(loc_A[loc_off]), GET_REAL(full_A[i*m+j]));
          loc_off++;
        }
      }
    }
    if (myRank == 0) printf("Verification of ScaLAPACK QR completed\n");
  }
  
  time = MPI_Wtime();
  for (iter=0; iter<niter; iter++){
    for (i=0; i<m*n/numPes; i++){
      loc_A[i] = drand48();
    }
    cpxgeqrf<dtype>(m,n,loc_A,1,1,desc_A,loc_TAU2,work,lwork,&info);
  }
  time = MPI_Wtime()-time;
  
  dtype avg_qr_time = time/niter;

  if(myRank == 0){
    printf("Completed " PRId64 " iterations of QR\n", iter);
    printf("Scalapack QR (pgeqrf) on a " PRId64 "-by-" PRId64 " matrix took %lf seconds/iteration, at %lf GFlops\n",
            m, n, time/niter, (2.*m*n*n-(2./3.)*n*n*n)/(time/niter)*1.E-9);
  }
 
  if (tdorgqr){ 
    time = MPI_Wtime();
    for (iter=0; iter<niter; iter++){
      for (i=0; i<m*n/numPes; i++){
        loc_A[i] = drand48();
      }    
      cpxorgqr<dtype>(m,n,MIN(m,n),loc_A,1,1,desc_A,loc_TAU2,work,lwork,&info);
    }
    time = MPI_Wtime()-time;
    
    dtype avg_formq_time = time/niter;
    
    if(myRank == 0){
      printf("Completed " PRId64 " iterations of QR\n", iter);
      //printf("Scalapack form Q (porgqr) from a " PRId64 "-by-" PRId64 " matrix took %lf seconds/iteration, at %lf GFlops\n",
      //       m, n, time/niter, (2.*m*n*n-(2./3.)*n*n*n)/(time/niter)*1.E-9);
      printf("Scalapack form Q (porgqr) from a " PRId64 "-by-" PRId64 " matrix took %lf seconds/iteration\n",
              m, n, time/niter);
      printf("Total SCALAPCK time to compute and form Q is %lf\n",
              avg_qr_time+avg_formq_time);
    }
  }


}


int main(int argc, char **argv) {
  int myRank, numPes, tdorgqr, bcomplex;
  int64_t n, b, niter, pr, pc, ipr, ipc, i, j, loc_off, m, verify, nz;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  if (myRank == 0){
    printf("%s [number of rows] [number of columns] [blocking factor] [number of iterations] [verify]", argv[0]);
    printf(" [num processor rows] [whether to benchmark form Q] [use complex]\n");
  }
  if (argc > 9){
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

    if (argc > 3) b = atoi(argv[3]);
  else b = 10;
  if (argc > 4) niter = atoi(argv[4]);
  else niter = 5;
  if (argc > 5) verify = atoi(argv[5]);
  else verify = 1;
  if (argc > 6) pr = atoi(argv[6]);
  else pr = numPes;
  if (argc > 7) tdorgqr = atoi(argv[7]);
  else tdorgqr = 1;
  if (argc > 8) bcomplex = atoi(argv[8]);
  else bcomplex = 0; 
  pc = numPes / pr;
  if (numPes % pr != 0) {
    if (myRank == 0){
      printf("" PRId64 " mod " PRId64 " != 0 Number of processor grid ", n, pc);
      printf("columns must divide into the matrix dimension\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  ipc = myRank / pr;
  ipr = myRank % pr;
 
  if (argc > 1) m = atoi(argv[1]);
  else m = b*10*pr;
  if (argc > 2) n = atoi(argv[2]);
  else n = b*10*pc;
  
  if (numPes % pr != 0) {
    if (myRank == 0){
      printf("%d mod " PRId64 " != 0 Number of processor grid ", numPes, pr);
      printf("rows must divide into number of processors\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  if (m % pr != 0) {
    if (myRank == 0){
      printf("" PRId64 " mod " PRId64 " != 0 Number of processor grid ", n, pr);
      printf("rows must divide into the matrix dimension\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
 
  
  if (myRank == 0){ 
    if (tdorgqr)
      printf("Benchmarking ScaLAPACK QR and form Q of ");
    else
      printf("Benchmarking ScaLAPACK QR of ");
    if (bcomplex) printf("a complex<double> ");
    else printf("a double precision ");
    printf("" PRId64 "-by-" PRId64 " matrix with block size " PRId64 "\n",m,n,b);
    printf("Using %d processors in " PRId64 "-by-" PRId64 " grid.\n", numPes, pr, pc);
  }
  if (bcomplex)
    bench_scala_qr< std::complex<double> >(m, n, b, niter, verify, pr, tdorgqr);
  else
    bench_scala_qr<double>(m, n, b, niter, verify, pr, tdorgqr);
  MPI_Finalize();
  return 0;
} /* end function main */


