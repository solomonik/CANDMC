/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <assert.h>
#include <string>
#include <fstream>
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

// Note: this generic, non-template-specialized class function will not be used, and it does not use the file string that is all-important now. See the specialized class and its function below.
template <typename dtype>
inline void bench_scala_qr(int64_t m, int64_t n, int64_t b, int64_t niter, int64_t verify, int64_t pr, int tdorgqr, std::string& fptrString){
dtype * loc_A, * full_A, * work;
  abort();
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


// Note: I had to make this template specialization for T=double to get lwork to work correctly.
template <>
inline void bench_scala_qr<double>(int64_t m, int64_t n, int64_t b, int64_t niter, int64_t verify, int64_t pr, int tdorgqr, std::string& fptrString){
  double * loc_A, * full_A, * work;
  int myRank, numPes;
  int64_t pc, ipr, ipc, i, j, loc_off, nz;
  double* TAU2, *loc_TAU2;
  int64_t* iwork, *iclustr, *ifail;
  int icontxt, iam, inprocs;
  int64_t iter, lwork;
  int info;
  char cC = 'C';
  int desc_A[9], desc_EC[9];
  double time;
  int64_t localDiv = m/pr;

  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  pc = numPes / pr;
  ipc = myRank / pr;
  ipr = myRank % pr;

  std::string fptrStrTotalNoFormQ = fptrString + "_NoFormQ.txt";
  std::string fptrStrTotalFormQ = fptrString + "_FormQ.txt";
  std::ofstream fptrTotalNoFormQ,fptrTotalFormQ;
  if (myRank == 0)
  {
    fptrTotalNoFormQ.open(fptrStrTotalNoFormQ.c_str());
    fptrTotalFormQ.open(fptrStrTotalFormQ.c_str());
  }

  int64_t size1 = m; size1*=n; size1 /= numPes; size1 *= sizeof(double);
  loc_A    = (double*)malloc(size1);
  if (verify){
    int64_t size2 = size1; size2 * numPes;
    full_A  = (double*)malloc(size2);
    TAU2  = (double*)malloc(n*sizeof(double));
  }

  srand48(666);

  Cblacs_pinfo(&iam,&inprocs);
  Cblacs_get(-1, 0, &icontxt);
  Cblacs_gridinit(&icontxt, &cC, pr, pc);
  cdescinit(desc_A, m, n,
		        b, b,
		        0, 0,
		        icontxt, localDiv, 
				    &info);
//  printf("Info - %d\n",info);
  assert(info==0);
  // first query for the correct size of lwork
  double sizeQuery;
  loc_TAU2  = (double*)malloc(n*sizeof(double)/pc);
  cpxgeqrf<double>(m,n,loc_A,1,1,desc_A,loc_TAU2,&sizeQuery,-1,&info);
  
  lwork       = sizeQuery;
  work        = (double*)malloc(lwork*sizeof(double));
  if (myRank == 0)
  {
    printf("lwork - %d\n", lwork);
  }

  int64_t size2=m; size2 *= n; size2 *= n;
  if ((verify) && (size2 < 1E9)){
     if (myRank == 0) printf("Verifying cpgeqrf correctness\n");
    init_matrices(m,n,ipr,pr,ipc,pc,b,full_A,loc_A); 
    cxgeqrf<double>(m,n,full_A,m,TAU2,work,lwork,&info);
    cpxgeqrf<double>(m,n,loc_A,1,1,desc_A,loc_TAU2,work,lwork,&info);
    loc_off = 0;
    for (i=0; i<n; i++){
      for (j=0; j<m; j++){
        if ((i/b)%pc == ipc && (j/b)%pr == ipr){
          int64_t size3 = i; size3 *= m;
          double diff = full_A[size3+j]-loc_A[loc_off];
          if (GET_REAL(diff)/GET_REAL(full_A[size3+j]) > 1.E-6)
            printf("incorrect answer " PRId64 " scalapack computed %E, lapack computed %E\n",
                    i, GET_REAL(loc_A[loc_off]), GET_REAL(full_A[size3+j]));
          loc_off++;
        }
      }
    }
    if (myRank == 0) printf("Verification of ScaLAPACK QR completed\n");
  }
  
  double totalTime=0;
  for (iter=0; iter<niter; iter++)
  {
    int64_t size3 = m; size3 *= n; size3  /= numPes;
    if (myRank == 0) {printf("Rank 0 size3 - %d, size1 - %d\n", size3, size1);}
    for (i=0; i<size3; i++){
      loc_A[i] = drand48();
    }
    double iterStartTime,iterTimeLocal,iterTimeGlobal;
    MPI_Barrier(MPI_COMM_WORLD);
    if (myRank == 0) {printf("Every process reached barrier.\n");}
    if (myRank == 0) printf("Rank %d has args - %d, %d, %d, %d, and descA stuff - %d %d %d %d %d %d %d %d %d", myRank, m, n, lwork, info, desc_A[0], desc_A[1], desc_A[2], desc_A[3], desc_A[4], desc_A[5], desc_A[6], desc_A[7], desc_A[8]);
    iterStartTime=MPI_Wtime();
    cpxgeqrf<double>(m,n,loc_A,1,1,desc_A,loc_TAU2,work,lwork,&info);
    if (myRank == 0) printf("Process %d done with QR routine with info param - %d\n", myRank, info);
    iterTimeLocal=MPI_Wtime()-iterStartTime;
    MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myRank == 0)
    {
      printf("No Form Q, numPes - %d, iter - %d, m - %d, n - %d, time - %g\n", numPes, iter, m, n, iterTimeGlobal);
      fptrTotalNoFormQ << numPes << "\t" << iter << "\t" << m << "\t" << n << "\t" << iterTimeGlobal << std::endl;
    }
    totalTime += iterTimeGlobal;
  }
  
  double avg_qr_time = totalTime/niter;

  if(myRank == 0){
    int64_t size3 = m; size3 *= n; size3 *= n;
    int64_t size4 = n; size4 *= n; size4 *= n;
    printf("Completed " PRId64 " iterations of QR\n", iter);
    printf("Scalapack QR (pgeqrf) on a " PRId64 "-by-" PRId64 " matrix took %lf seconds/iteration, at %lf GFlops\n",
            m, n, totalTime/niter, (2.*size3-(2./3.)*size4)/(totalTime/niter)*1.E-9);
  }
 
  if (tdorgqr){
    totalTime=0;
    for (iter=0; iter<niter; iter++)
    {
      int64_t size3 = m; size3 *= n; size3 /= numPes;
      for (i=0; i<size3; i++){
        loc_A[i] = drand48();
      }
      double iterStartTime,iterTimeLocal,iterTimeGlobal;
      if (myRank == 0) printf("before\n");
      iterStartTime=MPI_Wtime();
      cpxorgqr<double>(m,n,MIN(m,n),loc_A,1,1,desc_A,loc_TAU2,work,lwork,&info);
      iterTimeLocal=MPI_Wtime()-iterStartTime;
      if (myRank == 0) printf("after\n");
      MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      if (myRank == 0)
      {
        printf("Form Q, numPes - %d, iter - %d, m - %d, n - %d, time - %g\n", numPes, iter, m, n, iterTimeGlobal);
        fptrTotalFormQ << numPes << "\t" << iter << "\t" << m << "\t" << n << "\t" << iterTimeGlobal << std::endl;
      }
      totalTime += iterTimeGlobal;
    }
    
    double avg_formq_time = totalTime/niter;
    
    if(myRank == 0){
      printf("Completed " PRId64 " iterations of QR\n", iter);
      //printf("Scalapack form Q (porgqr) from a " PRId64 "-by-" PRId64 " matrix took %lf seconds/iteration, at %lf GFlops\n",
      //       m, n, time/niter, (2.*m*n*n-(2./3.)*n*n*n)/(time/niter)*1.E-9);
      printf("Scalapack form Q (porgqr) from a " PRId64 "-by-" PRId64 " matrix took %lf seconds/iteration\n",
              m, n, totalTime/niter);
      printf("Total SCALAPCK time to compute and form Q is %lf\n",
              avg_qr_time+avg_formq_time);
    }
  }

  if (myRank == 0)
  {
    fptrTotalNoFormQ.close();
    fptrTotalFormQ.close();
  }
}


int main(int argc, char **argv) {
  int myRank, numPes, tdorgqr, bcomplex;
  int64_t n, b, niter, pr, pc, ipr, ipc, i, j, loc_off, m, verify, nz;
  std::string fptrString;  

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  if (myRank == 0){
    printf("%s [number of rows] [number of columns] [blocking factor] [number of iterations] [verify]", argv[0]);
    printf(" [num processor rows] [whether to benchmark form Q] [use complex] [file string base]\n");
  }
  if (argc > 10){
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
      //printf("" PRId64 " mod " PRId64 " != 0 Number of processor grid ", n, pc);
      printf("columns must divide into the matrix dimension\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  if (argc > 9) fptrString=argv[9];
  else fptrString="noNameScalaQRbenchmark";

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
    bench_scala_qr< std::complex<double> >(m, n, b, niter, verify, pr, tdorgqr, fptrString);
  else
    bench_scala_qr<double>(m, n, b, niter, verify, pr, tdorgqr, fptrString);
  MPI_Finalize();
  return 0;
} /* end function main */


