#ifndef __TOPO_PDGEMM_ALGS_H__
#define __TOPO_PDGEMM_ALGS_H__

#include "../../shared/comm.h"

typedef struct ctb_args {
  char  trans_A;
  char  trans_B;
  int64_t   n;
  int64_t lda_A;
  int64_t lda_B;
  int64_t lda_C;
  int64_t buffer_size;
  int ovp;
} ctb_args_t;

void summa(ctb_args_t const * args,
           double const * mat_A,
           double const * mat_B,
           double     * mat_C,
           double   * buffer,
           CommData_t   cdt_row,
           CommData_t   cdt_col);

void d25_summa(ctb_args_t const  * args,
        double    * mat_A,
        double    * mat_B,
        double    * mat_C,
        double    * buffer,
#ifdef USE_MIC
		int			mic_portion,
		int			mic_id,
#endif
        CommData_t      cdt_row,
        CommData_t      cdt_col,
        CommData_t      cdt_kdir);

void d25_summa_ovp(ctb_args_t const  * args,
      double      * mat_A,
      double      * mat_B,
      double      * mat_C,
      double      * buffer,
#ifdef USE_MIC
	  int			mic_portion,
	  int			mic_id,
#endif
      CommData_t      cdt_row,
      CommData_t      cdt_col,
      CommData_t      cdt_kdir);

void bcast_cannon_4d(ctb_args_t const * args,
         double   * mat_A,
         double     * mat_B,
         double     * mat_C,
         double   * buffer,
         CommData_t     cdt_x1,
         CommData_t     cdt_y1,
         CommData_t     cdt_x2,
         CommData_t     cdt_y2);


#endif


