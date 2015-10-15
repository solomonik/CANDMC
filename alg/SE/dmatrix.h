#ifndef __DMATRIX_H__
#define __DMATRIX_H__

/**
 * \brief distributed matrix class which wraps scalapack
 */
class DMatrix{
public:
  //scalapack descriptor
  int desc[9];
  //number of rows in global matrix
  int64_t nrow;
  //number of columns in global matrix
  int64_t ncol;
  //block size
  int64_t b;
  //lda of local matrix chunk
  int64_t lda;
  //2D processor grid context
  pview pv;
  //local data pointer
  double * data;
  //allocated data pointer
  double * root_data;
  //whether the tensor is transposed (whether scalapack calls should use 'T')
  bool is_transp;
  //whether to free buffer on deletion of object, set to
  // true if buffer allocation done in constructor
//  bool free_buffer;
  char * name;

  /**
   * \brief default constructor
   */
  DMatrix();
  
  /**
   * \brief destructor frees data if free_buffer is true
   */
  //~DMatrix();
  /**
   * \brief destructor frees data and name
   */
  void destroy();

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
  DMatrix(int64_t           nrow_, 
          int64_t           ncol_,
          int64_t           b_,
          pview const &     pv_,
          int64_t           lda_ = 0,
          double *          root_data_ = NULL,
          double *          data_ = NULL,
          bool              is_transp_=false);

  /**
   * \brief constructor with name
   *
   * \param[in] nrow_ number of rows in global matrix
   * \param[in] ncol_ number of columns in global matrix
   * \param[in] b_ (distribution) block size of matrix
   * \param[in] pv_ 2D processor grid context
   * \param[in] name string name of tensor
   */
  DMatrix(int64_t           nrow_, 
          int64_t           ncol_,
          int64_t           b_,
          pview const &     pv_,
          char const *      name_);

  // \brief set name in a way that keeps compilers happy
  void set_name(char const * name_);


  /**
   * \brief DMatrix object can be cast to double * (by reference) to access data
   */
  operator double*() const;
  
  /**
   * \brief prints DMatrix data to stdout
   */
  void print();
 
  /**
   * \brief computes then number of local rows based on root processor
   */ 
  int64_t get_mynrow();

  /**
   * \brief computes then number of local cols based on root processor
   */ 
  int64_t get_myncol();

  /**
   * \brief computes the size of the local block based on root processor
   */ 
  int64_t get_mysize();

  /**
   * \brief obtains a moves the head pointer in data, changes rows and cols
   * \param[in] mrow number of rows to move by
   * \param[in] mcol number of cols to move by
   */
  void move_ptr(int64_t mrow,
                int64_t mcol);
  
  /**
   * \brief moves pointer to submatrix (without copying data)
   * \param[in] firstrow first global row to give slice
   * \param[in] numrows number of rows in sliced global submatrix
   * \param[in] firstcol first global col to give slice
   * \param[in] numcols number of cols in sliced global submatrix
   */
  void move_ptr(int64_t firstrow,
                int64_t numrows,
                int64_t firstcol,
                int64_t numcols);

  /**
   * \brief returns a transposed matrix with unchanged data layout (sets is_transp)
   */
  DMatrix transp();

  /**
   * \brief returns a transposed matrix with transposed data (sets is_transp)
   */
  DMatrix transpose_data();

  /**
   * \brief returns a matrix in a 1D layout on each processor row
   */
  double * replicate_vertical();

  /**
   * \brief returns a matrix in a 1D layout on each processor column
   */
  double * replicate_horizontal();

  /**
 * \brief performs a reduction from a column-replicated 1D layout to the 2D data layout of this matrix, adding to existing data
   * \param[in,out] cntrb piece of local data to contribute to reduciton, buffer used, contains garbage on output
   */
  void reduce_scatter_horizontal(double * cntrb);



  /**
   * \brief obtains a slice of the matrix by reference (without copying data)
   * \param[in] firstrow first global row to give slice
   * \param[in] numrows number of rows in sliced global submatrix
   * \param[in] firstcol first global col to give slice
   * \param[in] numcols number of cols in sliced global submatrix
   */
  DMatrix slice(int64_t firstrow,
                int64_t numrows,
                int64_t firstcol,
                int64_t numcols);

  /**
   * \brief sets active part of data buffer to zero
   */
  void set_to_zero();

  /**
   * \brief adds a scaled matrix to this matrix (this += alpha*A)
   *        requires that the local chunks of this and A are the same size
   * \param[in] A matrix to add
   * \param[in] alpha scaling factor for A
   */
  void daxpy(DMatrix  A,
             double   alpha);

  /**
   * \brief extracts active local submatrix into a contiguous slice
   * \return new global matrix with new data pointer and copied data
   */
  DMatrix get_contig();


  /**
   * \brief clones the matrix into a new object without setting data
   * \return new global matrix with unset data
   */
  DMatrix clone();

  /**
   * \brief transforms the matrix from nrow-by-ncol to nrow*factor-by-ncol/factor
   *        by reordering the local data 
   * \param[in] factor see description of function
   * \return new tensor with same but reordered local data
   */
  DMatrix foldcols(int64_t factor);
     
  /**
   * \brief transforms the matrix from nrow-by-ncol to nrow/factor-by-ncol*factor
   *        by reordering the local data 
   * \param[in] factor see description of function
   * \return new tensor with same but reordered local data
   */
  DMatrix foldrows(int64_t factor);

  /**
   * \brief transforms the matrix from 3D replicated to 2D rectangular layout
   * \param[in] pv3d 3D processor grid context
   * \return new tensor with same but reordered local data
   */
  DMatrix scatter_rows(pview_3d const * pv3d);

  /**
   * \brief zero upper triangle (keep diagonal)
   */
  void zero_upper_tri();

  /**
   * \brief zero lower triangle (keep diagonal)
   */
  void zero_lower_tri();

  /**
   * \brief scale the diagonal and add something to it
   * \param[in] scale scaling factor for existing diagonal
   * \param[in] add additive factor to add to diagonal after scaling
   */
  void diag_scale_add(double scale, double add);

  /**
   * \brief obtain character string of the transpose state for lapack/scalapack calls
   */
  char get_tp();
};

/**
 * \brief computes a QR and stores the output (Y,R) as distributed matrices, 
 *        can computes invT from Y
 */
class DQRMatrix {
public:
  //upper-triangular (square buffer) R factor from QR 
  DMatrix R;
  //lower-triangular unit-diagonal (rect buffer) Y factor from QR 
  DMatrix Y;


  /**
   * \brief constructor immediately computes QR
   * \param[in] A matrix to compute the QR of
   */
  DQRMatrix(DMatrix   A);

  ~DQRMatrix();
  
  /**
   * \brief computes inverse T from Y
   * \return distributed matrix corresponding to inverse of T
   */
  DMatrix compute_invT();
};

/**
 * \brief simplified wrapper for distributed gemm C=beta*C+alpha*A*B
 */
void cpdgemm(DMatrix    A, 
             DMatrix    B, 
             double     alpha, 
             DMatrix    C,
             double     beta);

/**
 * \brief simplified wrapper for distributed syrk C=beta*C+alpha*A'*A
 */
void cpdsyrk(DMatrix    A, 
             double     alpha, 
             DMatrix    C,
             double     beta);

/**
 * \brief simplified wrapper for distributed trsm B=alpha*A^-1*B
 */
void cpdtrsm(DMatrix    A,
             double     alpha,
             DMatrix    B);

#endif
