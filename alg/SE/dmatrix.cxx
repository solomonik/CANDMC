/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */
/* Author: Edgar Solomonik, June 16, 2014 */

/* File contains routines for reduction for distributed matrix wrapper */

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "../shared/util.h"
#include "dmatrix.h"
#include "../QR/qr_2d/qr_2d.h"

//#define PRINTALL
#define PRINTOFF

/**
 * \brief default constructor
 */
DMatrix::DMatrix(){ 
  data = NULL;
  root_data = NULL;
  name = NULL;
}
  
/**
 * \brief destructor frees data if free_buffer is true
 */
//DMatrix::~DMatrix(){
 // if (free_buffer) free(data);
//}
void DMatrix::destroy(){
  free(root_data);
  if (name != NULL) free(name);
}

/**
 * \brief main constructor
 *
 * \param[in] nrow_ number of rows in global matrix
 * \param[in] ncol_ number of columns in global matrix
 * \param[in] b_ (distribution) block size of matrix
 * \param[in] pv_ 2D processor grid context
 * \param[in] lda_ lda of local data chunk
 * \param[in] root_data_ corner data pointer
 * \param[in] data_ local data pointer
 * \param[in] is_transp_ whether the tensor is transposed 
 *                      (whether scalapack calls should use 'T')
 */
DMatrix::DMatrix(int64_t           nrow_, 
                 int64_t           ncol_,
                 int64_t           b_,
                 pview const &     pv_,
                 int64_t           lda_,
                 double *          root_data_,
                 double *          data_,
                 bool              is_transp_) {

  int info;
  nrow =nrow_;
  ncol = ncol_;
  //printf("created matrix with %d rows and %d cols\n", nrow, ncol);
  b = b_;
  pv = pv_;
  is_transp = is_transp_;
  if (lda_ != 0)
    lda = lda_;
  else
    lda = get_mynrow(); //((nrow/b)/pv.ccol.np+(((nrow/b)%pv.ccol.np)>0))*b;
  if (data_ == NULL){
    data = (double*)malloc(sizeof(double)*get_myncol()*lda);
    root_data = data;
   // free_buffer = true;
  } else {
    data = data_;
    root_data = root_data_;
 //   free_buffer = false;
  }
  name = NULL;
  cdescinit(desc, nrow, ncol,
            b, b,
            pv.rrow, pv.rcol,
            pv.ictxt, lda, 
            &info);
}


/**
 * \brief constructor with name
 *
 * \param[in] nrow_ number of rows in global matrix
 * \param[in] ncol_ number of columns in global matrix
 * \param[in] b_ (distribution) block size of matrix
 * \param[in] pv_ 2D processor grid context
 * \param[in] name string name of tensor
 */
DMatrix::DMatrix(int64_t           nrow_, 
                 int64_t           ncol_,
                 int64_t           b_,
                 pview const &     pv_,
                 char const *      name_){
  int info;
  nrow =nrow_;
  ncol = ncol_;
  b = b_;
  pv = pv_;
  is_transp = 0;
  lda = get_mynrow(); 
  data = (double*)malloc(sizeof(double)*get_myncol()*lda);
  root_data = data;
  name = (char*)malloc(strlen(name_)+1);
  strcpy(name, name_);
  cdescinit(desc, nrow, ncol,
            b, b,
            pv.rrow, pv.rcol,
            pv.ictxt, lda, 
            &info);

}

// \brief set name in a way that keeps compilers happy
void DMatrix::set_name(char const * name_){
  if (name_ != NULL){
    name = (char*)malloc(strlen(name_)+1);
    strcpy(name, name_);
  }
}


/**
 * \brief DMatrix object can be cast to double * (by reference) to access data
 */
DMatrix::operator double*() const {
  return data;
}

/**
 * \brief prints DMatrix data to stdout
 */
void DMatrix::print(){
#ifndef PRINTOFF
  if (pv.cworld.rank == 0){
    if (name == NULL)
      printf("Printing %d-by-%d distributed matrix laid out on %d-by-%d processor grid\n", 
             nrow, ncol, pv.ccol.np, pv.crow.np);
    else
      printf("Printing %s %d-by-%d distributed matrix laid out on %d-by-%d processor grid\n", 
             name, nrow, ncol, pv.ccol.np, pv.crow.np);
  }
  int myrow = 0;
  int mycol = 0;
  double sval[b];
  double rval[b];
  MPI_Request req;
  for (int row=0; row<nrow; row++){
    for (int col=0; col<ncol; col+=b){
      MPI_Barrier(pv.cworld.cm);
      if ((row/b)%pv.ccol.np == pv.ccol.rank &&
          (col/b)%pv.crow.np == pv.crow.rank){
        for (int icol=0; icol<b; icol++){
          //printf("processor %d lda= %d b = %d printing %d %d \n", pv.cworld.rank,lda,b,myrow,mycol);
          sval[icol] =  data[myrow+mycol*lda];
          mycol++;
          //printf("processor done %d printing %d %d \n", pv.cworld.rank,myrow,mycol);
        }
        MPI_Isend(sval, b, MPI_DOUBLE, 0, 133, pv.cworld.cm, &req);
      }
      if (pv.cworld.rank == 0){
        MPI_Recv(rval, b, MPI_DOUBLE, ((row/b)%pv.ccol.np)+pv.ccol.np*((col/b)%pv.crow.np), 133, pv.cworld.cm, MPI_STATUS_IGNORE);
        for (int icol=0; icol<b; icol++){
          printf("%+.2lf ", rval[icol]);
        }
      }
      if ((row/b)%pv.ccol.np == pv.ccol.rank &&
          (col/b)%pv.crow.np == pv.crow.rank){
        MPI_Wait(&req,MPI_STATUS_IGNORE);
      }
    }
    mycol = 0;
    if ((row/b)%pv.ccol.np == pv.ccol.rank)
      myrow++;
    MPI_Barrier(pv.cworld.cm);
    if (pv.cworld.rank == 0){
      printf("\n");
    }
  }
#endif
}
  
/**
 * \brief computes then number of local rows based on root processor
 */ 
int64_t DMatrix::get_mynrow(){
  return ((nrow/b)/pv.ccol.np+(((nrow/b)%pv.ccol.np)>(pv.ccol.rank + pv.ccol.np - pv.rrow)%pv.ccol.np))*b;
}

/**
 * \brief computes then number of local cols based on root processor
 */ 
int64_t DMatrix::get_myncol(){
  return ((ncol/b)/pv.crow.np+(((ncol/b)%pv.crow.np)>(pv.crow.rank + pv.crow.np - pv.rcol)%pv.crow.np))*b;
}

/**
 * \brief computes the size of the local block based on root processor
 */ 
int64_t DMatrix::get_mysize(){
  return get_mynrow()*get_myncol();
}


/**
 * \brief obtains a DMatrix with a moved head pointer in data
 * \param[in] mrow number of rows to move by
 * \param[in] mcol number of cols to move by
 * \return DMatrix with fewer rows and cols pointing to offset data pointer in this matrix
 */
void DMatrix::move_ptr(int64_t mrow,
                       int64_t mcol){
  DMatrix slc = slice(mrow, nrow-mrow, mcol, ncol-mcol);
  *this = slc;
}
  
/**
 * \brief moves pointer to submatrix (without copying data)
 * \param[in] firstrow first global row to give slice
 * \param[in] numrows number of rows in sliced global submatrix
 * \param[in] firstcol first global col to give slice
 * \param[in] numcols number of cols in sliced global submatrix
 */
void DMatrix::move_ptr(int64_t firstrow,
                       int64_t numrows,
                       int64_t firstcol,
                       int64_t numcols){
  DMatrix slc = slice(firstrow, numrows, firstcol, numcols);
  *this = slc;
}

/**
 * \brief returns a transposed matrix (sets is_transp)
 */
DMatrix DMatrix::transp(){
  DMatrix mat = DMatrix(nrow, ncol, b, pv, lda, root_data, data, !is_transp);
  mat.set_name(name);
  return mat;
}

/**
 * \brief returns a transposed matrix with transposed data (sets is_transp)
 */
DMatrix DMatrix::transpose_data(){
  DMatrix mat = get_contig();
  int64_t mnrow = mat.get_mynrow();
  int64_t mncol = mat.get_myncol();
  DMatrix mat2 = get_contig();
  MPI_Sendrecv(mat2, mnrow*mncol, MPI_DOUBLE, pv.crow.rank+pv.ccol.rank*pv.crow.np, 342,
               mat, mnrow*mncol, MPI_DOUBLE, pv.crow.rank+pv.ccol.rank*pv.crow.np, 342,
               pv.cworld.cm, MPI_STATUS_IGNORE);
  mat.set_name(name);
  mat2.destroy();
  return mat;
}

/**
 * \brief returns a matrix in a 1D layout on each processor row
 */
double * DMatrix::replicate_vertical(){
//  CommData_t cself;
//  cself.np = 1;
//  cself.rank = 0;
//  cself.cm = MPI_COMM_SELF;
//  pview rep_view;
//  rep_view.rrow = 0;
//  rep_view.rcol = pv.rcol;
//  rep_view.crow = pv.crow;
//  rep_view.ccol = cself;
//  rep_view.ictxt = -1;  

  DMatrix ctg = get_contig(); 

  double * rep_mat = (double*)malloc(sizeof(double)*nrow*get_myncol());

  MPI_Allgather(ctg, ctg.get_mysize(), MPI_DOUBLE,
                rep_mat, ctg.get_mysize(), MPI_DOUBLE,
                pv.ccol.cm);

  return rep_mat;
}

/**
 * \brief returns a matrix in a 1D layout on each processor column
 */
double * DMatrix::replicate_horizontal(){
  DMatrix ctg = get_contig(); 

  double * rep_mat = (double*)malloc(sizeof(double)*ncol*get_mynrow());

  MPI_Allgather(ctg, ctg.get_mysize(), MPI_DOUBLE,
                rep_mat, ctg.get_mysize(), MPI_DOUBLE,
                pv.crow.cm);

  return rep_mat;
}

/**
 * \brief performs a reduction from a column-replicated 1D layout to the 2D data layout of this matrix, adding to existing data
 * \param[in,out] cntrb piece of local data to contribute to reduciton, buffer used, contains garbage on output
 */
void DMatrix::reduce_scatter_horizontal(double * cntrb){
  int jp=1;
  int logdep = 0;
  while (jp<pv.crow.np){
    jp*=2;
    logdep++;
  }
  assert(jp==pv.crow.np);
  int64_t cntrb_size = ncol*get_mynrow();

  double * recv_buf = (double*)malloc(sizeof(double)*cntrb_size/2);

  double * ptr_keep = cntrb;
  double * ptr_send;
  int ghost_rank = 0;
  for (int i=0; i<logdep; i++){
    int64_t chunk_size = cntrb_size/(2<<i);
    int nbr;
    if ((pv.crow.rank >> i) % 2 == 1){
      //data will end up in the wrong place at the end since butterfly is inverted
      // so keep track of effective rank and transpose at the end
      // butterfly is inverted so that large sends travel fewer hops
      ghost_rank += 1<<(logdep-i-1);
      nbr = pv.crow.rank - (1<<i);
      ptr_send = ptr_keep;
      ptr_keep = ptr_keep+chunk_size;
    } else {
      nbr = pv.crow.rank + (1<<i);
      ptr_send = ptr_keep + chunk_size;
    }

    MPI_Sendrecv(ptr_send, chunk_size, MPI_DOUBLE, nbr, i,
                 recv_buf, chunk_size, MPI_DOUBLE, nbr, i,
                 pv.crow.cm, MPI_STATUS_IGNORE);
    
    cdaxpy(chunk_size, 1.0, recv_buf, 1, ptr_keep, 1);
  }
  if (ghost_rank != pv.crow.rank){
    MPI_Sendrecv(ptr_keep, get_mysize(), MPI_DOUBLE, ghost_rank, 17,
                 ptr_send, get_mysize(), MPI_DOUBLE, ghost_rank, 17,
                 pv.crow.cm, MPI_STATUS_IGNORE);
  } else ptr_send = ptr_keep;
  assert(lda == get_mynrow());
  cdaxpy(get_mysize(), 1.0, ptr_send, 1, data, 1);
  free(recv_buf);
}




/**
 * \brief obtains a slice of the matrix by reference (without copying data)
 * \param[in] firstrow first global row to give slice
 * \param[in] numrows number of rows in sliced global submatrix
 * \param[in] firstcol first global col to give slice
 * \param[in] numcols number of cols in sliced global submatrix
 */
DMatrix DMatrix::slice(int64_t firstrow,
                       int64_t numrows,
                       int64_t firstcol,
                       int64_t numcols){
  char pname[120];
  strcpy(pname, "slice ");
  if (name != NULL)
    strcat(pname, name);
  CTF_Timer tmr(pname);
  tmr.start();

  pview pvr = pv;
  LIBT_ASSERT(firstrow % b == 0);
  LIBT_ASSERT(firstcol % b == 0);
  pvr.rrow = (pvr.rrow + (firstrow/b))%pv.ccol.np;
  pvr.rcol = (pvr.rcol + (firstcol/b))%pv.crow.np;
  DMatrix dfslc = DMatrix(nrow-firstrow, ncol-firstcol, b, pvr, lda, root_data, data, is_transp);
  dfslc.data +=  get_mynrow() - dfslc.get_mynrow();
  dfslc.data += (get_myncol() - dfslc.get_myncol())*lda;
  DMatrix dslc = DMatrix(numrows, numcols, b, pvr, lda, dfslc.root_data, dfslc.data, is_transp);
  dslc.set_name(name);

  tmr.stop();
  return dslc;
}

/**
 * \brief sets active part of data buffer to zero
 */
void DMatrix::set_to_zero(){
  char pname[120];
  strcpy(pname, "set_to_zero ");
  if (name != NULL)
    strcat(pname, name);
  CTF_Timer tmr(pname);
  tmr.start();


  int64_t mnrow = get_mynrow();
  int64_t mncol = get_myncol();
  for (int64_t i = 0; i<mncol; i++){
    std::fill(data+lda*i, data+lda*i+mnrow, 0.0);
  }
  tmr.stop();
}

/**
 * \brief adds a scaled matrix to this matrix (this += alpha*A)
 *        requires that the local chunks of this and A are the same size
 * \param[in] A matrix to add
 * \param[in] alpha scaling factor for A
 */
void DMatrix::daxpy(DMatrix  A,
                    double   alpha){
  char pname[120];
  strcpy(pname, "daxpy ");
  if (name != NULL)
    strcat(pname, name);
  CTF_Timer tmr(pname);
  tmr.start();

  LIBT_ASSERT(A.get_mynrow() == get_mynrow());
  LIBT_ASSERT(A.get_myncol() == get_myncol());
  int64_t mnrow = get_mynrow();
  int64_t mncol = get_myncol();
  int lda_A = A.lda;
  int str_A = 1;
  double * ptr_A = A;
  if (A.is_transp){  
    lda_A = 1;
    str_A = A.lda;
    ptr_A = (double*)malloc(sizeof(double)*mnrow*mncol);
    MPI_Sendrecv(A, mnrow*mncol, MPI_DOUBLE, pv.crow.rank+pv.ccol.rank*pv.crow.np, 342,
                 ptr_A, mnrow*mncol, MPI_DOUBLE, pv.crow.rank+pv.ccol.rank*pv.crow.np, 342,
                 pv.cworld.cm, MPI_STATUS_IGNORE);

  }
  int lda_B = lda;
  int str_B = 1;
  double * ptr_B = data;
  if (is_transp){
    lda_B = 1;
    str_B = lda;
    ptr_B = (double*)malloc(sizeof(double)*mnrow*mncol);
    MPI_Sendrecv(data, mnrow*mncol, MPI_DOUBLE, pv.crow.rank+pv.ccol.rank*pv.crow.np, 342,
                 ptr_B, mnrow*mncol, MPI_DOUBLE, pv.crow.rank+pv.ccol.rank*pv.crow.np, 342,
                 pv.cworld.cm, MPI_STATUS_IGNORE);
  }
  //printf("lda_A = %d, str_A = %d, lda_B = %d, str_B = %d\n", lda_A, str_A, lda_B, str_B);
  for (int64_t i=0; i<mncol; i++){
    cdaxpy(mnrow, alpha, ptr_A+lda_A*i, str_A, ptr_B+lda_B*i, str_B);
  }
  if (A.is_transp)
    free(ptr_A);
  if (is_transp)
    free(ptr_B);
  tmr.stop();
}

/**
 * \brief extracts active local submatrix into a contiguous slice
 * \return new global matrix with new data pointer and copied data
 */
DMatrix DMatrix::get_contig(){
  if (get_mynrow() == lda){
    //printf("got my contig nrow = %d, ncol = %d\n",nrow,ncol);
    DMatrix ctg = this->clone();
    ctg.set_to_zero();
    ctg.daxpy(*this,1.0);
    return ctg;
  } else {
    DMatrix ctg = DMatrix(nrow, ncol, b, pv, get_mynrow(), NULL, NULL, is_transp);
    //printf("copying %d by %d block with lda %d to lda %d\n", nrow, ncol, lda, get_mynrow());
    lda_cpy(get_mynrow(), get_myncol(), lda, get_mynrow(), data, (double*)ctg);
    return ctg;
  }
}

/**
 * \brief clones the matrix into a new object without setting data
 * \return new global matrix with unset data
 */
DMatrix DMatrix::clone(){
  DMatrix ctg = DMatrix(nrow, ncol, b, pv, get_mynrow(), NULL, NULL, is_transp);
  ctg.set_name(name);
  return ctg;
}
/**
 * \brief transforms the matrix from 3D replicated to 2D rectangular layout
 * \param[in] pv3d 3D processor grid context
 * \return new tensor with same but reordered local data
 */
DMatrix DMatrix::scatter_rows(pview_3d const * pv3d){
  int64_t mncol = get_myncol();

  MPI_Barrier(pv3d->cworld.cm);
  DMatrix drect = DMatrix(nrow, ncol, b, pv3d->prect);
  drect.set_to_zero();
  CPRINTF(pv3d->cworld, "drect:\n");
  MPI_Barrier(pv3d->cworld.cm);
  MPI_Barrier(pv3d->prect.cworld.cm);
  drect.print();
  printf("allgather clyr rank %d my rank %d\n", pv3d->clyr.rank, pv3d->cworld.rank);
  int64_t rmnrow = drect.get_mynrow();
  int64_t nlrows[pv3d->clyr.np];
  MPI_Allgather(&rmnrow, 1, MPI_INT64_T, nlrows, 1, MPI_INT64_T, pv3d->clyr.cm);
  printf("done allgather clyr rank %d\n", pv3d->clyr.rank);
  int pfx = 0;
  for (int i=0; i<pv3d->clyr.rank; i++){
    pfx += nlrows[i];
  }
  lda_cpy(rmnrow, mncol, lda, rmnrow, data+pfx, drect.data);
  return drect;
}
/**
 * \brief transforms the matrix from nrow-by-ncol to nrow*factor-by-ncol/factor
 *        by reordering the local data 
 * \param[in] factor see description of function
 * \return new tensor with same but reordered local data
 */
DMatrix DMatrix::foldcols(int64_t factor){
  char pname[120];
  strcpy(pname, "foldcols ");
  if (name != NULL)
    strcat(pname, name);
  CTF_Timer tmr(pname);
  tmr.start();


  int64_t mnrow = get_mynrow();
  int64_t mncol = get_myncol();
  LIBT_ASSERT(mnrow % (b*factor) == 0);
  int64_t brow = mnrow / factor;
  //FIXME copies some extra stuff
  DMatrix ctg = get_contig();
  for (int64_t j=0; j<mnrow/(b*factor); j++){
    for (int64_t i=0; i<factor; i++){
      lda_cpy(b, mncol, lda, mnrow/factor, 
              data+i*b+j*b*factor, ctg.data+i*brow*mncol+j*b);
    }
  }
  tmr.stop();
  DMatrix ret= DMatrix(ctg.nrow/factor,ctg.ncol*factor,b,pv,ctg.lda/factor,ctg.root_data,ctg.data,ctg.is_transp);
  ret.set_name(name);
  return ret;
} 

/**
 * \brief transforms the matrix from nrow-by-ncol to nrow/factor-by-ncol*factor
 *        by reordering the local data 
 * \param[in] factor see description of function
 * \return new tensor with same but reordered local data
 */
DMatrix DMatrix::foldrows(int64_t factor){
  char pname[120];
  strcpy(pname, "foldrows ");
  if (name != NULL)
    strcat(pname, name);
  CTF_Timer tmr(pname);
  tmr.start();

  int64_t mncol = get_myncol();
  int64_t mnrow = get_mynrow();
  LIBT_ASSERT(mncol % factor == 0);
  int64_t bcol = mncol / factor;
  DMatrix ctg = get_contig();
  //FIXME: loop over rows with each new b rows coming from a different factor
  for (int64_t j=0; j<mnrow/b; j++){
    for (int64_t i=0; i<factor; i++){
      lda_cpy(b, bcol, lda, mnrow*factor, 
              data+i*bcol*lda+j*b, ctg.data+i*b+j*b*factor);
    }
  }
  tmr.stop();
  DMatrix ret = DMatrix(ctg.nrow*factor,ctg.ncol/factor,b,pv,ctg.lda*factor,ctg.root_data,ctg.data,ctg.is_transp);
  ret.set_name(name);
  return ret;
} 

/**
 * \brief zero upper triangle (keep diagonal)
 */
void DMatrix::zero_upper_tri(){
  int64_t row_offset = ((pv.ccol.rank + pv.ccol.np - pv.rrow)%pv.ccol.np)*b;
  int64_t col_offset = ((pv.crow.rank + pv.crow.np - pv.rcol)%pv.crow.np)*b;
  double * data_ptr;// = data;
  //printf("nrow = %d, ncol = %d, lda = %d\n", nrow, ncol, lda);
  int64_t lcol = 0;
  for (int64_t col=col_offset; col<ncol; col+=b*pv.crow.np, lcol+=b){
    int64_t lrow = 0;
    for (int64_t row=row_offset; row<nrow; row+=b*pv.ccol.np, lrow+=b){
      data_ptr = data + lcol*lda + lrow;
      //zero upper triangle and diagonal of block if its on the diagonal
      if (row == col){
        for (int i=0; i<b; i++){
          std::fill(data_ptr+i*lda, data_ptr+i*(1+lda), 0.0);
        }
      }
      //zero whole block if its above the diagonal
      if (row < col){
        //printf("row = %d, col = %d sizeof(double) = %d pointer moved by %d\n",row,col,sizeof(double),data_ptr-data);
        for (int i=0; i<b; i++){
          std::fill(data_ptr+i*lda, data_ptr+i*lda+b, 0.0);
        }
      }
    }
  }
}

/**
 * \brief zero lower triangle (keep diagonal)
 */
void DMatrix::zero_lower_tri(){
  int64_t row_offset = ((pv.ccol.rank + pv.ccol.np - pv.rrow)%pv.ccol.np)*b;
  int64_t col_offset = ((pv.crow.rank + pv.crow.np - pv.rcol)%pv.crow.np)*b;
  double * data_ptr;
  int64_t lcol = 0;
  for (int64_t col=col_offset; col<ncol; col+=b*pv.crow.np, lcol+=b){
    int64_t lrow = 0;
    for (int64_t row=row_offset; row<nrow; row+=b*pv.ccol.np, lrow+=b){
      data_ptr = data + lcol*lda + lrow;
      //zero lower triangle of block if its on the diagonal
      if (row == col){
        for (int i=0; i<b; i++){
          std::fill(data_ptr+i*(1+lda)+1, data_ptr+i*lda+b, 0.0);
        }
      }
      //zero whole block if its below the diagonal
      if (row > col){
        for (int i=0; i<b; i++){
          std::fill(data_ptr+i*lda, data_ptr+i*lda+b, 0.0);
        }
      }
    }
  }
}

/**
 * \brief scale the diagonal and add something to it
 * \param[in] scale scaling factor for existing diagonal
 * \param[in] add additive factor to add to diagonal after scaling
 */
void DMatrix::diag_scale_add(double scale, double add){
  int64_t row_offset = ((pv.ccol.rank + pv.ccol.np - pv.rrow)%pv.ccol.np)*b;
  int64_t col_offset = ((pv.crow.rank + pv.crow.np - pv.rcol)%pv.crow.np)*b;
  double * data_ptr;
  int64_t lcol = 0;
  for (int64_t col=col_offset; col<ncol; col+=b*pv.crow.np, lcol+=b){
    int64_t lrow = 0;
    for (int64_t row=row_offset; row<nrow; row+=b*pv.ccol.np, lrow+=b){
      data_ptr = data + lcol*lda + lrow;
      if (row == col){
        for (int i=0; i<b; i++){
          data_ptr[i*(1+lda)] = data_ptr[i*(1+lda)] * scale + add;
        }
      }
    }
  }
}

/**
 * \brief obtain character string of the transpose state for lapack/scalapack calls
 */
char DMatrix::get_tp(){
  if (is_transp) return 'T';
  else return 'N';
}

/**
 * \brief constructor immediately computes QR
 * \param[in] A matrix to compute the QR of
 */
DQRMatrix::DQRMatrix(DMatrix A){
  /*char pname[120];
  strcpy(pname, "2D QR ");
  if (A.name != NULL)
    strcat(pname, A.name);
  CTF_Timer tmr(pname);
  tmr.start();
*/
  Y = A.get_contig();
  pview pv_cpy = Y.pv;
  //QR_2D(Y.data, Y.lda, Y.nrow, Y.ncol, Y.b, &pv_cpy, NULL, 0);
  //QR_scala_2D(Y.data, Y.lda, Y.nrow, Y.ncol, Y.b, &pv_cpy, NULL, 0, Y.desc, Y, 1, 1);
  QR_2D_pipe(Y.data, Y.lda, Y.nrow, Y.ncol, Y.b, &pv_cpy, NULL, 0, NULL, NULL);
  //printf("slicing %d by %d R\n", Y.ncol, Y.ncol);
  R = Y.slice(0,Y.ncol,0,Y.ncol).get_contig();
  R.zero_lower_tri();
  Y.zero_upper_tri();
  Y.diag_scale_add(0.0, 1.0);
  Y.print();
  R.print();
  //tmr.stop();
}

DQRMatrix::~DQRMatrix(){
  Y.destroy();
  R.destroy();
}

/**
 * \brief computes inverse T from Y
 * \return distributed matrix corresponding to inverse of T
 */
DMatrix DQRMatrix::compute_invT(){
  DMatrix T = R.clone();
  T.set_name("T");
  cpdsyrk(Y, 1.0, T, 0.0);
  T.zero_upper_tri();
  T.diag_scale_add(.5, 0.0);
  return T;
}

/**
 * \brief simplified wrapper for distributed gemm C=beta*C+alpha*A*B
 */
void cpdgemm(DMatrix    A, 
             DMatrix    B, 
             double     alpha, 
             DMatrix    C,
             double     beta){
  char pname[120];
  strcpy(pname, "cpdgemm ");
  if (C.name != NULL)
    strcat(pname, C.name);
  else
    strcat(pname, "?");
  strcat(pname, " <- ");
  if (A.name != NULL)
    strcat(pname, A.name);
  else
    strcat(pname, "?");
  strcat(pname, " * ");
  if (B.name != NULL)
    strcat(pname, B.name);
  else
    strcat(pname, "?");
  CTF_Timer tmr(pname);
  tmr.start();

  LIBT_ASSERT(C.is_transp == false);
  int64_t k = A.ncol;
  if (A.is_transp) k = A.nrow;
  //printf("A.nrow = %d, B.nrow = %d, C.nrow = %d\n",A.desc[2],B.desc[2],C.desc[2]);
  //printf("A.ncol = %d, B.ncol = %d, C.ncol = %d\n",A.desc[3],B.desc[3],C.desc[3]);
  cpdgemm(A.get_tp(), B.get_tp(), C.nrow, C.ncol, k, alpha, A, 1, 1, A.desc,
          B, 1, 1, B.desc, beta, C, 1, 1, C.desc);
  tmr.stop();
} 

/**
 * \brief simplified wrapper for distributed syrk C=beta*C+alpha*A'*A
 */
void cpdsyrk(DMatrix    A, 
             double     alpha, 
             DMatrix    C,
             double     beta){
  char pname[120];
  strcpy(pname, "cpdsyrk ");
  if (C.name != NULL)
    strcat(pname, C.name);
  else 
    strcat(pname,"?");
  strcat(pname,"<-");
  if (A.name != NULL)
    strcat(pname, A.name);
  else 
    strcat(pname,"?");
  strcat(pname,"^2");
  CTF_Timer tmr(pname);
  tmr.start();

  LIBT_ASSERT(C.is_transp == false);
  int K;
  if (!A.is_transp) K = A.nrow;
  else K = A.ncol;
  LIBT_ASSERT(!A.is_transp);
  cpdsyrk('L', 'T', C.nrow, K, alpha, A, 1, 1, A.desc,
          beta, C, 1, 1, C.desc);
  tmr.stop();
}

/**
 * \brief simplified wrapper for distributed trsm B=alpha*A^-1*B
 */
void cpdtrsm(DMatrix    A,
             double     alpha,
             DMatrix    B){
  char pname[120];
  strcpy(pname, "cpdtrsm ");
  if (A.name != NULL)
    strcat(pname, A.name);
  strcat(pname, " ");
  if (B.name != NULL)
    strcat(pname, B.name);
  CTF_Timer tmr(pname);
  tmr.start();

  LIBT_ASSERT(A.is_transp == false);
  LIBT_ASSERT(B.is_transp == false);
  cpdtrsm('R', 'L', 'N', 'N', B.nrow, B.ncol, alpha, A,
          1, 1, A.desc, B, 1, 1, B.desc);
  tmr.stop();
}


