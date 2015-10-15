#ifndef __LAPACK_H__
#define __LAPACK_H__

#include "util.h"

#ifndef LAPACKHASTSQR
#define LAPACKHASTSQR 0
#endif

void cdgemm(char transa,        char transb,
            int m,              int n,
            int k,              double a,
            const double * A,   int lda,
            const double * B,   int ldb,
            double b,           double * C,
                                int ldc);

void cdtrsm(char    SIDE,	  char            UPLO,
		        char    TRANSA,	char            DIAG,
		        int     M,		  int             N,
		        double  alpha,	const double  * A,
		        int     lda,	  double        * B,
					  int     ldb);

void cdgetrf(int      M,    int N,
		         double * A,		int lda,
		         int * IPIV,		int * info);
   
void cdsyrk(char const          UPLO,
            char const          TRANS,
            int const           M,
            int const           K,
            double const        ALPHA,
            double const *      A,
            int const           LDA,
            double const        BETA,
            double *            C,
            int const           LDC);

void cdgeqrf(int const          M,
             int const          N,
             double     *       A,
             int const          LDA,
             double     *       TAU2,
             double     *       WORK,
             int const          LWORK,
             int        *       INFO);
template<typename dtype>
void cxgeqrf(int const          M,
             int const          N,
             dtype     *       A,
             int const          LDA,
             dtype     *       TAU2,
             dtype     *       WORK,
             int const          LWORK,
             int        *       INFO);


void cdormqr(char const         SIDE,
             char const         TRANS,
             int const          M,
             int const          N,
             int const          K,
             double const *     A,
             int const          LDA,
             double const *     TAU2,
             double     *       C,
             int const          LDC,
             double     *       WORK,
             int const          LWORK,
             int        *       INFO);

double cdlange(char const       NORM,
               int const        M,
               int const        N,
               double const *   A,
               int const        LDA,
               double   *       WORK);

void cdger(int const            M,
           int const            N,
           double const         alpha,
           const double *       X,
           const int            incX,
           const double *       Y,      
           const int            incY,
           double *             A,
           const int            lda_A);


void cdorgqr(int const          M,
             int const          N,
             int const          K,
             double     *       A,
             int const          LDA,
             double const  *       TAU2,
             double     *       WORK,
             int const          LWORK,
             int        *       INFO);

void cdtrtri(char     uplo,
             char     diag,
             int      m,
             double * T,
             int      lda_T,
             int *    info);

void cdlacpy(char uplo, int m, int n, double const * A, int lda_A, double * B, int lda_B);

    

void czgemm(char transa,  char transb,
            int m,        int n,
            int k,        std::complex<double> a,
            const std::complex<double> * A,     int lda,
            const std::complex<double> * B,     int ldb,
            std::complex<double> b,       std::complex<double> * C,
                                int ldc);

template <typename dtype>
void cxgemm(char transa,  char transb,
            int m,        int n,
            int k,        dtype a,
            const dtype * A,    int lda,
            const dtype * B,    int ldb,
            dtype b,      dtype * C,
                                int ldc);

void cdaxpy(int n,        double dA,
            const double * dX,  int incX,
            double * dY,        int incY);

void czaxpy(int n,                        std::complex<double> dA,
            const std::complex<double> * dX,    int incX,
            std::complex<double> * dY,          int incY);

void cdcopy(int n,
            const double * dX,  int incX,
            double * dY,        int incY);

void czcopy(int n,
            const std::complex<double> * dX,    int incX,
            std::complex<double> * dY,          int incY);

void cdscal(int n,        double dA,
            double * dX,  int incX);

void czscal(int n,        std::complex<double> dA,
            std::complex<double> * dX,  int incX);

template <typename dtype>
void cxaxpy(int n,        dtype dA,
            const dtype * dX,   int incX,
            dtype * dY, int incY);

template <typename dtype>
void cxcopy(int n,
            const dtype * dX,   int incX,
            dtype * dY, int incY);

template <typename dtype>
void cxscal(int n, dtype dA,
            dtype * dX,  int incX);

double cddot(int n,       const double *dX,
             int incX,    const double *dY,
             int incY);

#if (LAPACKHASTSQR==1)
void cdtpqrt(int const         m,
             int const         n,
             int const         l,
             int const         nb,
             double *          A,
             int const         lda_A,
             double *          B,
             int const         lda_B,
             double *          T,
             int const         lda_T,
             double *          work,
             int *             info);

void cdtpmqrt( char const  side,
               char const  trans,
               int const   m,
               int const   n,
               int const   k,
               int const   l,
               int const   NB,
               double *    V,
               int const   lda_V,
               double *    T,
               int const   lda_T,
               double *    A,
               int const   lda_A,
               double *    B,
               int const   lda_B,
               double *    work,
               int *       info);
                
void cdtprfb(char const  side,
             char const  trans,
             char const  DIRECT,
             char const  STOREV,
             int const   m,
             int const   n,
             int const   k,
             int const   l,
             double *    V,
             int const   lda_V,
             double *    T,
             int const   lda_T,
             double *    A,
             int const   lda_A,
             double *    B,
             int const   lda_B,
             double *    work,
             int         ldwork);
#endif

void cdlarft(char           f, 
             char           c, 
             int            m, 
             int            b, 
             double const * Y, 
             int            lda_Y, 
             double const * tau,
             double *       T, 
             int            lda_T);

void cdlarfb(char           l, 
             char           t, 
             char           f, 
             char           c, 
             int            m, 
             int            k, 
             int            b, 
             double const * Y, 
             int            lda_Y, 
             double const * T, 
             int            lda_T,
             double *       B, 
             int            lda_B, 
             double *       buffer, 
             int            buf_sz);

void cdlatrd(char      UPLO,
             int       N,
             int       NB,
             double *  A,
             int       LDA,
             double *  E,
             double *  TAU,
             double *  W,
             int       LDW);

void cdsytrd(char     UPLO,
             int      N,
             double * A,
             int      LDA,
             double * D,
             double * E,
             double * TAU,
             double * WORK,
             int      LWORK,
             int *    INFO);

void cdsyevx(char     JOBZ, 
             char     RANGE, 
             char     UPLO, 
             int      N, 
             double * A, 
             int      LDA, 
             double   VL, 
             double   VU, 
             int      IL,
             int      IU, 
             double   ABSTOL, 
             int *    M, 
             double * W, 
             double * Z, 
             int      LDZ, 
             double * WORK, 
             int      LWORK, 
             int *    IWORK, 
             int *    IFAIL, 
             int *    INFO);

void cdgetri(int      N,
             double * A,
             int      LDA,
             int *    IPIV,
             double * WORK,
             int      lwork,
             int  *   info);

#ifdef USE_SCALAPACK
extern "C" {
  void Cblacs_pinfo(int*, int*);
  void Cblacs_get(int, int, int*);
  void Cblacs_gridinit(int*, char*, int, int);
  void Cblacs_gridinfo(int, int*, int*, int*, int*);
  void Cblacs_gridmap(int*, int*, int, int, int);
  void Cblacs_barrier(int , char*);
  void Cblacs_gridexit(int);
}
  
void cpdgemm( char n1,     char n2,
              int sz1,     int sz2,
              int sz3,     double ALPHA,
              double * A,  int ia,
              int ja,      int * desca,
              double * B,  int ib,
              int jb,      int * descb,
              double BETA, double * C,
              int ic,      int jc,
                           int * descc);
 
void cpdsyrk(char  UPLO,
             char  TRANS,
             int    N,
             int    K,
             double  ALPHA,
             double const *  A,
             int  IA,
             int  JA,
             int *   DESCA,
             double  BETA,
             double *  C,
             int   IC,
             int   JC,
             int *   DESCC);

void cpdtrsm(char   SIDE,
             char   UPLO,
             char   TRANS,
             char   DIAG,
             int    M,
             int    N,
             double   ALPHA,
             double const *  A,
             int    IA,
             int    JA,
             int *   DESCA,
             double *  B,
             int    IB,
             int    JB,
             int *   DESCB);

void cpdgeqrf(int  M,
              int  N,
              double *  A,
              int  IA,
              int  JA,
              int const *     DESCA,
              double *  TAU2,
              double *  WORK,
              int  LWORK,
              int *     INFO);
  
template <typename dtype>
void cpxgeqrf(int  M,
              int  N,
              dtype *  A,
              int  IA,
              int  JA,
              int const *     DESCA,
              dtype *  TAU2,
              dtype *  WORK,
              int  LWORK,
              int *     INFO);



void cdescinit(int * desc, 
               const int m,	    const int n,
               const int mb, 	  const int nb,
               const int irsrc,	const int icsrc,
               const int ictxt,	const int LLD,
               int * info);
 
void cpdlatrd( char      UPLO,
               int       N,
               int       NB,
               double *  A,
               int       IA,
               int       JA,
               int *     DESCA,
               double *  D,
               double *  E,
               double *  TAU,
               double *  W,
               int       IW,
               int       JW,
               int *     DESCW,
               double *  WORK);


void cpdorgqr(int  M,
              int  N,
              int  K,
              double *  A,
              int  IA,
              int  JA,
              int const *     DESCA,
              double *  TAU2,
              double *  WORK,
              int  LWORK,
              int *     INFO);

template <typename dtype>
void cpxorgqr(int  M,
              int  N,
              int  K,
              dtype *  A,
              int  IA,
              int  JA,
              int const *     DESCA,
              dtype *  TAU2,
              dtype *  WORK,
              int  LWORK,
              int *     INFO);
 

void cpdsytrd( char     UPLO,
               int      N,
               double * A,
               int      IA,
               int      JA,
               int *    DESCA,
               double * D,
               double * E,
               double * TAU,
               double * WORK,
               int      LWORK,
               int *    INFO);

void cpdsyevx( char     JOBZ, 
               char     RANGE, 
               char     UPLO, 
               int      N, 
               double * A, 
               int      IA, 
               int      JA, 
               int *    DESCA, 
               double   VL, 
               double   VU, 
               int      IL,
               int      IU, 
               double   ABSTOL, 
               int *    M, 
               int *    NZ, 
               double * W, 
               double   ORFAC, 
               double * Z, 
               int      IZ, 
               int      JZ, 
               int *    DESCZ,
               double * WORK, 
               int      LWORK, 
               int *    IWORK, 
               int      LIWORK, 
               int *    IFAIL, 
               int *    ICLUSTR, 
               double * GAP,
               int *    INFO);

#endif

             

#endif
