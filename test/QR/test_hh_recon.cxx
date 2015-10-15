/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */
/* Parts of this test written by Mathias Jacquelin */

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
void hh_recon_unit_test(const char *  filename, 
                        int64_t const myRank, 
                        int64_t const numPes, 
                        int64_t const req_id, 
                        CommData_t    cdt){
#else
void hh_recon_unit_test(int64_t const m,
                        int64_t const b, 
                        int64_t const myRank, 
                        int64_t const numPes, 
                        int64_t const req_id, 
                        CommData_t    cdt){
#endif
  if (myRank == 0)  
    printf("unit testing parallel TSQR with YT reconstruction m=" PRId64 " b=" PRId64 " np=" PRId64 "...\n",m,b,numPes);
  double *A,*whole_A,*collect_YR,*whole_YR,*swork,*tau,*bw_A;
  double norm, yn2;
  int64_t i,j,row,col,mb,pass;
  int info;

  int64_t seed_offset = 99900;

#ifdef READ_FILE
  int64_t m,b;
  read_matrix(filename, myRank, numPes, whole_A, m, b, A, mb);
  assert(m%numPes == 0);

#else
  assert(m%numPes == 0);
  mb = m / numPes;
  assert(0==(posix_memalign((void**)&whole_A,
          ALIGN_BYTES,
          m*b*sizeof(double))));
  assert(0==(posix_memalign((void**)&A,
          ALIGN_BYTES,
          mb*b*sizeof(double))));


  for (col=0; col<b; col++){
    for (row=0; row<mb; row++){
      srand48(seed_offset + ((myRank*mb + row)+(col*m))*61);
      A[row+col*mb] = drand48();
/*      if (myRank*mb+row == col)
        A[row+col*mb] += 2*m;*/
    }
  }
  for (col=0; col<b; col++){
    for (row=0; row<m; row++){
      srand48(seed_offset + (row + col*m)*61);
      whole_A[row+col*m] = drand48();
/*      if (row == col)
        whole_A[row+col*m] += 2*m;*/
    }
  }
#endif

  assert(0==(posix_memalign((void**)&tau,
          ALIGN_BYTES,
          b*sizeof(double))));
  assert(0==(posix_memalign((void**)&swork,
          ALIGN_BYTES,
          m*b*sizeof(double))));
  assert(0==(posix_memalign((void**)&collect_YR,
          ALIGN_BYTES,
          m*b*sizeof(double))));
  assert(0==(posix_memalign((void**)&whole_YR,
          ALIGN_BYTES,
          m*b*sizeof(double))));
  assert(0==(posix_memalign((void**)&bw_A,
          ALIGN_BYTES,
          m*b*sizeof(double))));
  double * W;
  assert(0==(posix_memalign((void**)&W,
          ALIGN_BYTES,
          b*b*sizeof(double))));


  //get YR via TSQR and reconstruction
  hh_recon_qr(A,mb,m,b,W,myRank,numPes,0,0,cdt);

  MPI_Gather(A, mb*b, MPI_DOUBLE,
					   collect_YR, mb*b, MPI_DOUBLE, 0, cdt.cm);
  if (myRank == 0){
    for (i=0; i<numPes; i++){
      lda_cpy(mb, b, mb, m, collect_YR+mb*b*i, whole_YR+mb*i);
    }

    /*for (i=0; i<b; i++){
      tau[i] = 2./(1+cddot(m-i-1, whole_YR+m*i+i+1, 1, 
                                  whole_YR+m*i+i+1, 1));
//          tau[i] = cdlange('F',m-i-1,1, whole_YR + i+1 + i*m,m,NULL);
//          tau[i] = 2.0 / (pow(tau[i],2.0)+1.0);
//          tau[i] = (tau[i]==2.0)?0.0:tau[i];
    }*/
    tau_recon('L', b, m, m, whole_YR, tau);


    double * Q = (double*)malloc(m*m*sizeof(double));
    double * QQT = (double*)malloc(m*m*sizeof(double));
    double * tau_lapack = (double*) malloc(b*sizeof(double));


    if (m<=10){
      printf("TSQR Y/R is \n");
      print_matrix(whole_YR,m,b);
      printf("TSQR TAU is \n");
      print_matrix(tau,1,b);
    }



    //compute Q 
    /*lda_cpy(m,b,m,m,whole_A,bw_A);
    cdgeqrf(m,b,bw_A,m,tau,swork,m*b,&info);
    lda_cpy(m,b,m,m,bw_A,whole_YR);*/
    lda_cpy(m,b,m,m,whole_YR,Q);
    cdorgqr( m, m, b, Q, m, tau, QQT, m, &info );

    //A-QR check
    std::fill(bw_A, bw_A+m*b, 0.0);
    copy_upper(whole_YR, bw_A, b, m, m, 0);
    lda_cpy(m,b,m,m,whole_A,QQT);
    cdgemm('N','N',m,b,m,-1.0,Q,m,bw_A,m,1.0,QQT,m);

    /*for(i=0;i<b;i++){
      norm = cdlange('F',m,i+1, QQT,m,NULL);
      cout <<"TSQR   \t"<<norm<< "\t" << *(whole_YR+i+i*m) << endl;  
    }*/
    
    norm = cdlange('F',m,b, QQT,m,NULL);
    cout << "TSQR ||A - QR||_2 is "<< norm <<endl;


    //Do the same with LAPACK
    lda_cpy(m,b,m,m,whole_A,bw_A);
    cdgeqrf(m,b,bw_A,m,tau_lapack,swork,m*b,&info);


    if (m<=10){
      printf("LAPACK Y/R is \n");
      print_matrix(bw_A,m,b);
      printf("LAPACK TAU is \n");
      print_matrix(tau_lapack,1,b);
    }
    


    //compare the Ys
/*    copy_lower(whole_YR, swork, b,m, m, m, 1);
    std::fill(QQT, QQT+m*m, 0.0);
    copy_lower(bw_A, QQT, b,m, m, m, 1);
    for(i=0;i<b;i++){
      cdaxpy(m-i-1, -1.0, QQT+i+1+i*m, 1, swork+i+1+i*m, 1);
//      for(j=i+1;j<m;j++){
//        cout <<  *(swork+j+i*m) << endl;
//      }     

      norm = cdlange('F',m-i-1,1, swork + i*m,m,NULL);
      printf("||Y(:,%d) - Ylapack|| = %lE\n",i,norm);
    }

    norm = cdlange('F',m,b, swork,m,NULL);
    printf("||Y - Ylapack|| = %lE\n",norm);


    //Compare the Tau's
    cdaxpy(b, -1.0, tau_lapack, 1, tau, 1);
    cout << "-----Tau - Tau_lapack-----"<< endl;
    print_matrix(tau,b,1);
    cout << "--------------------------"<<endl;*/

    assert(info == 0);
    //get Q of LAPACK
    lda_cpy(m,b,m,m,bw_A,Q);
    cdorgqr( m, m, b, Q, m, tau_lapack, QQT, m, &info );
    /*printf("LAPACK first b columns of Q:\n");
    print_matrix(Q,m,b);*/
    //Get R of LAPACK
    std::fill(collect_YR, collect_YR+m*b, 0.0);
    copy_upper(bw_A, collect_YR, b, m, m, 0);
    
    lda_cpy(m,b,m,m,whole_A,QQT);
    cdgemm('N','N',m,b,m,-1.0,Q,m,collect_YR,m,1.0,QQT,m);

    /*for(i=0;i<b;i++){
      norm = cdlange('F',m,i+1, QQT,m,NULL);
      cout <<"LAPACK \t"<<norm<< "\t" << *(bw_A+i+i*m) << endl;  
    }*/

    norm = cdlange('F',m,b, QQT,m,NULL);
    cout << "LAPACK ||A - QR||_2 is "<< norm <<endl;

    /*for(i=0;i<b;i++){
      cout <<"R_LAPACK  - R_TSQR = \t"<<*(bw_A+i+i*m) - *(whole_YR+i+i*m)<< endl;  
    }*/


    free(tau_lapack);
    free(Q);
    free(QQT);


    std::fill(bw_A, bw_A+m*b, 0.0);
    copy_upper(whole_YR, bw_A, b, m, m, 0);
  

    cdormqr('L', 'N', m, b, b, whole_YR, m, tau, bw_A, 
            m, swork, m*b, &info);

//    print_matrix(whole_A, m, b);
    /*print_matrix(tau, 1, b);
    for (i=0; i<b; i++){
      yn2 = cddot(m-i-1, whole_A+m*i+i+1, 1, 
       whole_A+m*i+i+1, 1);
      tau[i] = 1.-(yn2-sqrt(2.-yn2))/(yn2+2);
      tau[i] = 2./(1+cddot(m-i-1, whole_A+m*i+i+1, 1, 
             whole_A+m*i+i+1, 1));
    }*/
  
    for (i=0; i<m*b; i++){
      bw_A[i] = whole_A[i] - bw_A[i];
    }
    /*print_matrix(bw_A, m, b);
    printf("\n\n");*/
    norm = 0.0;  
    for (i=0; i<b; i++){
      norm += sqrt(cddot(m, bw_A+i*m, 1, bw_A+i*m, 1)/cddot(m, whole_A+i*m, 1, whole_A+i*m, 1));
    }

    printf("||A-QR||_2 = %.2E\n", norm);
    
    if (norm > 1.E-9){
      printf("TEST FAILED!!!\n");
    } else {
      printf("Test successful.\n");
    }
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
  int64_t m, b;

  CommData_t cdt_glb;
  INIT_COMM(numPes, myRank, 1, cdt_glb);

#ifdef READ_FILE
  string filename;
  if (argc != 2 ) 
    printf("Usage: mpirun -np <num procs> ./exe <matrix file>\n");

  if (argc == 2) {
    filename.append(argv[1]);
  }
  hh_recon_unit_test(filename.c_str(), myRank, numPes, 0, cdt_glb);
  
#else
  if (argc != 3 && argc != 1) 
    printf("Usage: mpirun -np <num procs> ./exe <number of rows> <number of columns>\n");

  if (argc == 1) {
    b = 17;
    m = 34*numPes;
  }
  if (argc == 3) {
    m = atoi(argv[1]);
    b = atoi(argv[2]);
    assert(m > 0);
    assert(b > 0);
    assert(m % numPes == 0);
    assert(m / numPes >= b);
  }
  hh_recon_unit_test(m, b, myRank, numPes, 0, cdt_glb);
#endif  


  COMM_EXIT;
  return 0;
}
