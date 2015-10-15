/* Author: Edgar Solomonik, April 9, 2014 */

/* File contains routines for reduction from fully-dense to banded */

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "../shared/util.h"
#include "../QR/qr_2d/qr_2d.h"

//#define PRINTALL

/**
 * \brief Perform reduction to banded using 2D QR
 *
 * \param[in,out] A n-by-n dense symmetric matrix, stored unpacked
                    pointer should refer to current working corner of A
 * \param[in] lda_A lda of A
 * \param[in] n number of rows and columns in A
 * \param[in] b is the large block to which we are reducing the band, 
                b must be a multiple of b_sub
 * \param[in] b_sub small block size with which matrix is distributed
 * \param[in] pv current processor grid view oriented at corner of A 
 **/
void sym_full2band(double  * A,
                   int64_t   lda_A,
                   int64_t   n,
                   int64_t   b,
                   int64_t   b_sub,
                   pview   * pv){
  // if already within last square in band
  if (n<=b) return;

  TAU_FSTART(sym_full2band);

#ifdef PRINTALL
  if (pv->cworld.rank == 0)
    printf("A:\n");
  print_dist_mat(n, n, b_sub, pv->ccol.rank, pv->ccol.np, pv->rrow, 
                              pv->crow.rank, pv->crow.np, pv->rcol,
                 pv->cworld.cm, A, lda_A);
#endif

  // Assumpton: square processor grid
  assert(pv->crow.np == pv->ccol.np);

  // Assumption: block size divisibility
  assert(b%b_sub == 0 && b>=b_sub);

  // Transpose target proc
  int preflect = pv->crow.rank + pv->ccol.rank*pv->ccol.np;
  //printf("rank = %d, reflect rank = %d\n", pv->cworld.rank, preflect);

  MPI_Status stat;

  // offset of local pointer along matrix row
  int64_t loc_row_offset = b_sub*(b/(b_sub*pv->ccol.np));
  if ((pv->ccol.rank+pv->ccol.np-pv->rrow)%pv->ccol.np < (b/b_sub)%pv->ccol.np)
    loc_row_offset+=b_sub;
  
  // offset of local pointer along matrix col (multiplied by lda)
  int64_t loc_col_offset = lda_A*b_sub*(b/(b_sub*pv->crow.np));
  if ((pv->crow.rank+pv->crow.np-pv->rcol)%pv->crow.np < (b/b_sub)%pv->crow.np)
    loc_col_offset+=b_sub*lda_A;

  // number of local rows in the trailing matrix update
  int64_t mb = ((n-b)/b_sub)/pv->ccol.np;
  if ((pv->ccol.rank+pv->ccol.np-pv->rrow-((b/b_sub)%pv->ccol.np))%pv->ccol.np < ((n-b)/b_sub)%pv->ccol.np)
    mb++;
  mb *= b_sub;

  // number of local columns in the trailing matrix update
  int64_t kb = ((n-b)/b_sub)/pv->crow.np;
  if ((pv->crow.rank+pv->crow.np-pv->rcol-((b/b_sub)%pv->ccol.np))%pv->crow.np < ((n-b)/b_sub)%pv->crow.np)
    kb++;
  kb *= b_sub;

  
#ifdef PRINTALL
//  if (pv->cworld.rank == 0)
    printf("[%d] n=%d,b=%d,b_sub=%d,mb=%d,kb=%d\n",pv->cworld.rank,n,b,b_sub,mb,kb);
    printf("[%d] loc_row_offset = %d, loc_col_offset = %d\n", pv->cworld.rank, loc_row_offset, loc_col_offset);
#endif

  // rotate row, so that root row is correct for below 2D QR, 
  //  above determination of mb already accounted for adjusted root row
  pv->rrow = (pv->rrow+(b/b_sub)) % pv->ccol.np;
 
  double * Y = (double*)malloc(sizeof(double)*mb*b);
  std::fill(Y,Y+mb*b,0.0);

  // 2D QR on n-b by b rectangle, aggregate Y on each column 
  // FIXME: (could use smaller block size here and not aggregated Y)
  pview qr_pv = *pv;
  QR_2D(A+loc_row_offset, lda_A, n-b, b, b_sub, &qr_pv, Y, mb);

  // FIXME: this T was already computed within routine above
  // invT is replicated everywhere
  double * invT = (double*)malloc(sizeof(double)*b*b);
  std::fill(invT,invT+b*b,0.0);
  compute_invT_from_Y(Y, mb, invT, b, mb, b, pv);
#ifdef PRINTALL
  printf("invT:\n");
  for (int i=0; i<b; i++){
    for (int j=0; j<b; j++){
      printf("%+.2f ", invT[i+j*b]);
    }
    printf("\n");
  }
#endif
 
  // Compute W = (AY)' = Y'A'=Y'A is aggregated on each row
  double * W = (double*)malloc(sizeof(double)*kb*b);

  // Y replicated along rows [Y,Y], compute locally [A1*Y,A2*Y]
  TAU_FSTART(WeqAxY);
  if (mb != 0 && kb != 0)
    cdgemm('T','N',b,kb,mb,1.0,Y,mb,A+loc_row_offset+loc_col_offset,lda_A,0.0,W,b);
  else
    std::fill(W,W+kb*b,0.0);
  // Reduce within columns to diagonal to obtain [W1*Y, 0; 0, W2*Y]
  if (pv->ccol.rank == pv->crow.rank) {
    MPI_Reduce(MPI_IN_PLACE, W, kb*b, MPI_DOUBLE, MPI_SUM, pv->crow.rank, pv->ccol.cm);
  } else {
    MPI_Reduce(W, NULL,         kb*b, MPI_DOUBLE, MPI_SUM, pv->crow.rank, pv->ccol.cm);
  }
  TAU_FSTOP(WeqAxY)  

  // reuse buffer of Y for U
  double * U = Y;
  // reuse buffer of W for V'
  double * V = W;

  
  // Y is aggregated on each column and W is on the diagonal, so both are on diagonal
  TAU_FSTART(computeUandV);
  if (pv->ccol.rank == pv->crow.rank) {
    assert(kb==mb);
    int64_t lb = mb;
#ifdef PRINTALL
    if (pv->ccol.rank == 0)
      printf("Y:\n");
    print_dist_mat(n-b, b, b_sub, 
                   pv->cdiag.rank, pv->cdiag.np, pv->rrow,
                   0, 1, 0,   
                   pv->cdiag.cm, Y, lb);
    if (pv->ccol.rank == 0)
      printf("W:\n");
    print_dist_mat(b, n-b, b_sub, 0, 1, 0,   
                   pv->cdiag.rank, pv->cdiag.np, pv->rrow,  
                   pv->cdiag.cm, W, b);
#endif
    // Compute Z = Y'W' = Y'AY
    double * Z = (double*)malloc(sizeof(double)*b*b);
    if (lb != 0)
      cdgemm('T','T',b,b,lb,1.0,Y,lb,W,b,0.0,Z,b);
    else
      std::fill(Z,Z+b*b,0.0);
    MPI_Allreduce(MPI_IN_PLACE, Z, b*b, MPI_DOUBLE, MPI_SUM, pv->cdiag.cm);
#ifdef PRINTALL
    if (pv->ccol.rank == 0 && pv->crow.rank == 0){
      printf("Z:\n");
      print_matrix(Z, b,b);
    }
#endif
    
    if (lb != 0){
      // Compute U = YT' on the diagonal also,
      // since rather than T we actually have inv(T) we have a backsolve U*invT'=Y
      cdtrsm( 'R', 'L', 'N', 'N', lb, b, 1.0, invT, b, Y, lb);

      // no-op reminder
      U = Y;
#ifdef PRINTALL
      if (pv->ccol.rank == 0)
        printf("U:\n");
      print_dist_mat(n-b, b, b_sub,pv->cdiag.rank, pv->cdiag.np, pv->rrow, 0, 1, 0,
                     pv->cdiag.cm, U, lb);
#endif

      // Transform W yes into V' = W - .5ZU' = Y'A - .5Y'AYTY' 
      cdgemm('N','T',b,lb,b,-.5,Z,b,U,lb,1.0,W,b);
      
      // no-op reminder
      V = W;
    }
    free(Z);
#ifdef PRINTALL
    if (pv->ccol.rank == 0)
      printf("V:\n");
    print_dist_mat(b, n-b, b_sub, 
                   0,1,0,pv->cdiag.rank, pv->cdiag.np, pv->rrow,  
                   pv->cdiag.cm, V, b);
#endif
  }
  //Barrier for profiling purposes
  MPI_Barrier(pv->cworld.cm);
  TAU_FSTOP(computeUandV);
  TAU_FSTART(bcastUandV)
  // aggregate V' on each row
  MPI_Bcast(V, kb*b, MPI_DOUBLE, pv->crow.rank, pv->ccol.cm);
  // aggregate U on each column
  MPI_Bcast(U, mb*b, MPI_DOUBLE, pv->ccol.rank, pv->crow.cm);
  TAU_FSTOP(bcastUandV);
  
  // Compute UV'
  if (mb != 0 && kb != 0){
    double * UVT = (double*)malloc(sizeof(double)*mb*kb);
    TAU_FSTART(gemmUxV);
    cdgemm('N','N',mb,kb,b,1.0,U,mb,V,b,0.0,UVT,mb);
    TAU_FSTOP(gemmUxV);

    free(U);
    free(V);  

    // Transpose UV' to get VU'
    double * VUT = (double*)malloc(sizeof(double)*mb*kb);
    TAU_FSTART(sendrecvUxV);
    MPI_Sendrecv(UVT, mb*kb, MPI_DOUBLE, preflect, 0, 
                 VUT, kb*mb, MPI_DOUBLE, preflect, 0, pv->cworld.cm, &stat);
    TAU_FSTOP(sendrecvUxV);

#ifdef PRINTALL
    if (pv->cworld.rank == 0)
      printf("UVT:\n");
    print_dist_mat(n-b, n-b, b_sub, pv->ccol.rank, pv->ccol.np, pv->rrow, 
                                  pv->crow.rank, pv->crow.np, (pv->rcol+(b/b_sub))%pv->crow.np,
                   pv->cworld.cm, UVT, mb);

    if (pv->cworld.rank == 0)
      printf("A:\n");
    print_dist_mat(n-b, n-b, b_sub, pv->ccol.rank, pv->ccol.np, pv->rrow, 
                                  pv->crow.rank, pv->crow.np, (pv->rcol+(b/b_sub))%pv->crow.np,
                   pv->cworld.cm, A+loc_row_offset+loc_col_offset,lda_A);
#endif

    TAU_FSTART(AminusUxV);
    // Assumpton: square processor grid
    for (int64_t c=0; c<kb; c++){
      cdaxpy(mb, -1.0, UVT+c*mb, 1, A+loc_row_offset+loc_col_offset+c*lda_A, 1);
      cdaxpy(mb, -1.0, VUT+c, kb,   A+loc_row_offset+loc_col_offset+c*lda_A, 1);
      /*for (int64_t r=0; r<mb; r++){
        A[loc_row_offset+loc_col_offset+c*lda_A+r] -= UVT[c*mb+r] + VUT[r*kb+c];
      }*/
    }
    TAU_FSTOP(AminusUxV);
#ifdef PRINTALL
    if (pv->cworld.rank == 0)
      printf("A-UVT-VUT:\n");
    print_dist_mat(n-b, n-b, b_sub, pv->ccol.rank, pv->ccol.np, pv->rrow, 
                                  pv->crow.rank, pv->crow.np, (pv->rcol+(b/b_sub))%pv->crow.np,
                   pv->cworld.cm, A+loc_row_offset+loc_col_offset, lda_A);
#endif

  // Assumpton: square processor grid
    free(VUT);
    free(UVT);
  } else {
    free(U);
    free(V);
  }
 
  // root row rotated above 
  //  pv->rrow = (pv->rrow+1) % pv->ccol.np;

  // rotate root col

  pv->rcol = (pv->rcol+b/b_sub) % pv->crow.np;
  if (n-b > 0){
    sym_full2band(A+loc_col_offset+loc_row_offset,lda_A,n-b,b,b_sub,pv);
  }
#ifdef PRINTALL
  if (pv->cworld.rank == 0)
    printf("new A:\n");
  print_dist_mat(n, n, b_sub, pv->ccol.rank, pv->ccol.np, pv->rrow, 
                              pv->crow.rank, pv->crow.np, pv->rcol,
                 pv->cworld.cm, A, lda_A);
#endif
  TAU_FSTOP(sym_full2band);
}

