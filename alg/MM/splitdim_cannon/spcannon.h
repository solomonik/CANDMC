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

#ifndef __SPCANNON_H__
#define __SPCANNON_H__

#include "mpi.h"


void kput_cannon(int const      rank,
                 int const      kary,
                 int const      ndim,
                 MPI_Comm const comm,
                 int const      n,
                 int const      m,
                 int const      k,
                 char const     transp_A,
                 double const   alpha,
                 double *       A,
                 char const     transp_B,
                 double const   beta,
                 double *       B,
                 double *       C);

void kuni_cannon(int const      rank,
                 int const      kary,
                 int const      ndim,
                 MPI_Comm const comm,
                 int const      n,
                 int const      m,
                 int const      k,
                 char const     transp_A,
                 double const   alpha,
                 double *       A,
                 char const     transp_B,
                 double const   beta,
                 double *       B,
                 double *       C);


#endif// __SPCANNON_H__

