/* Copyright (c) 2011, Edgar Solomonik>
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following 
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL EDGAR SOLOMONIK BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY 
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF 
 * SUCH DAMAGE. */

#ifndef __VSPCANNON_H__
#define __VSPCANNON_H__

#include <vector>

#ifdef TAU
#include <Profile/Profiler.h>
#define TAU_FSTART(ARG)                                 \
    TAU_PROFILE_TIMER(timer##ARG, #ARG, "", TAU_USER);  \
    TAU_PROFILE_START(timer##ARG)

#define TAU_FSTOP(ARG)                                  \
    TAU_PROFILE_STOP(timer##ARG)

#else
#define TAU_PROFILE(NAME,ARG,USER)
#define TAU_PROFILE_TIMER(ARG1, ARG2, ARG3, ARG4)
#define TAU_PROFILE_STOP(ARG)
#define TAU_PROFILE_START(ARG)
#define TAU_FSTART(ARG)
#define TAU_FSTOP(ARG)
#endif
class ShftMsg : public CMessage_ShftMsg {
  public:
    double * data;
    int dim;
    int level;
    int pidx;
};

class StgrMsg : public CMessage_StgrMsg {
  public:
    double * data;
    int dim;
    int level;
};

class Main : public CBase_Main {
  public:
    int * dim_len;
    int np, iter, niter, nwarm, warmup, witer;
    int n, m, k;
    double st, end;
    double alpha, beta;

    Main(CkArgMsg* m);
    void done();
    void run_spc();
    void reduceC(CkReductionMsg * msg);
};

class Mapper : public CBase_Mapper {
  public:
    int np;
    int *mapping;

    Mapper(int ndim, int kary, int * dim_len);
    ~Mapper();
    int procNum(int, const CkArrayIndex &idx);
};


class VPblock : public CBase_VPblock {
  public:
    int n, m, k;
    double alpha, beta;
    int pidx, stgr_set, shft_set, level, nmsg;

    int * stgr_acc_table;
    int * shft_acc_table;
    int * shft_pidx;

    std::vector< ShftMsg* > shft_queue_A;
    std::vector< ShftMsg* > shft_queue_B;
    std::vector< StgrMsg* > stgr_queue_A;
    std::vector< StgrMsg* > stgr_queue_B;
  
    double *A, *B, *C;

    VPblock();
    VPblock(CkMigrateMessage *msg);
    
    void init_sindex(int nb, int mb, int kb);
    void init_rand(int n, int m, int k);
    void contract(double alpha, double beta);
    void staggerA(StgrMsg * msg);
    void staggerB(StgrMsg * msg);
    void stagger();
    void shiftA(ShftMsg * msg);
    void shiftB(ShftMsg * msg);
    void loc_shiftA(int im);
    void loc_shiftB(int im);
    void loc_staggerA(int im);
    void loc_staggerB(int im);
    void shift();
    void start_shift();
    void gatherC();
};

#endif// __VSPCANNON_H__

