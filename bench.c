/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/

#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#include "immintrin.h"

libxsmm_smmfunction sixteen3;
int M = 16;
int N = 16;
int K = 16;
int entries = 1;

LIBXSMM_INLINE void zero_buf(float* buf, long size) {
	  int i;
	    for (i = 0; i < size; ++i) {
		        buf[i] = 0.0f;
			  }
}

LIBXSMM_INLINE void init_buf(float* buf, long size, int initPos, int initOne)
{
	  int i;
	    zero_buf(buf, size);
	      for (i = 0; i < size; ++i) {
		          buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? drand48() : (0.05 - drand48()/10.0)));
			    }
}

int main(int argc, char* argv[])
{
  unsigned long long l_start, l_end, l_start2, l_end2;
  double l_total = 0.0, l_total2 = 0.0;
  double flops = 0.0;
  int i;
  int strategy;
  srand48(1);
  i = 1;
  if (argc > i ) M    = atoi(argv[i++]);
  if (argc > i) N     = atoi(argv[i++]);
  if (argc > i) K     = atoi(argv[i++]);
  if (argc > i) entries     = atoi(argv[i++]);
  if (argc > i) strategy    = atoi(argv[i++]);

  unsigned int prefetch = LIBXSMM_PREFETCH_AL1_BL1_CL1;


if ( strategy == 2 ) {
	  prefetch = LIBXSMM_PREFETCH_SIGONLY;          
} else if ( strategy == 3 ) {
	  prefetch = LIBXSMM_PREFETCH_BL2_VIA_C;          
} else if ( strategy == 4 ) {
	  prefetch = LIBXSMM_PREFETCH_AL2_AHEAD;          
} else if ( strategy == 5 ) {
	  prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD;          
} else if ( strategy == 6 ) {
	  prefetch = LIBXSMM_PREFETCH_AL2;          
} else if ( strategy == 7 ) {
	  prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C;          
} else if ( strategy == 8 ) {
	  prefetch = LIBXSMM_PREFETCH_AL2_JPST;          
} else if ( strategy == 9 ) {
	  prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST;          
} else if ( strategy == 11) {
	  prefetch = LIBXSMM_PREFETCH_AL1;          
} else if ( strategy == 12 ) {
	  prefetch = LIBXSMM_PREFETCH_BL1;          
} else if ( strategy == 13 ) {
	  prefetch = LIBXSMM_PREFETCH_CL1;          
} else if ( strategy == 14 ) {
	  prefetch = LIBXSMM_PREFETCH_AL1_BL1;          
} else if ( strategy == 15 ) {
	  prefetch = LIBXSMM_PREFETCH_BL1_CL1;          
} else if ( strategy == 16 ) {
	  prefetch = LIBXSMM_PREFETCH_AL1_CL1;          
} else if ( strategy == 17 ) {
	  prefetch = LIBXSMM_PREFETCH_AL1_BL1_CL1;          
} else {
	  prefetch = LIBXSMM_PREFETCH_NONE;          
}

  sixteen3 = libxsmm_smmdispatch(M, N, K, NULL, NULL, NULL, NULL, NULL, NULL, &prefetch );
  const int  iterations = 3000000;
  float *As  = (float*)libxsmm_aligned_malloc( 64 * M * K * (entries+65) * sizeof(float), 2097152);
  float *Bs  = (float*)libxsmm_aligned_malloc( 64 * K * N * (entries+65) * sizeof(float), 2097152);
  float *Cs  = (float*)libxsmm_aligned_malloc( 64 * M * N * (entries+65) * sizeof(float), 2097152);

  init_buf( As, 64 * M * K * (entries+65), 0 , 0);
  init_buf( Bs, 64 * K * N * (entries+65), 0 , 0);
  init_buf( Cs, 64 * M * N * (entries+65), 0 , 0);
  l_start2 = libxsmm_timer_tick();
  double shared_total = 0.0;
  double ext_flops;

  printf("Starting BENCHMARK...\n");

# pragma omp parallel firstprivate(As, Bs, Cs, iterations, entries, l_start, l_end, l_total, flops)
{
 l_total = 0;
 int my_id = omp_get_thread_num();
 const int MK = M*K, KN= K*N, MN = M*N ;
 float *A = (float*)  &As[my_id * (entries) * MK];
 float *B = (float*)  &Bs[my_id * (entries) * KN];
 float *C = (float*)  &Cs[my_id * (entries) * MN];

#ifdef ERROR_CHECK
  float *C_check = (float*) libxsmm_aligned_malloc( M*N *sizeof(float) , 2097152  );
  int mi, mj, mk;
  for ( mi = 0; mi < M; mi++   ) {
    for ( mj = 0; mj < N; mj++  ) {
      C_check[mj*M+mi] = C[ mj*M+mi  ];
    }
  }
#endif

 int it, loc = 0;
 int ii,ij;
 float *a, *b, *c, *a_p, *b_p, *c_p;
 a = A;
 b = B;
 c = C;
 a_p = A+MK;
 b_p = B+KN;
 c_p = C+MN;
 float *BOUND = A + entries * MK;

 if (my_id == 0)  {
   l_start = libxsmm_timer_tick();
 }
  for (it = 0; it < iterations; it++) {

   sixteen3(a, b, c, a_p, b_p, c_p);
   a += MK;
   b += KN;
   c += MN;
   a_p += MK;
   b_p += KN;
   c_p += MN;
   if (a == BOUND) {a = A;  b = B; c = C; }
   if (a_p == BOUND) {a_p = A;  b_p = B; c_p = C; }
 }

 if (my_id == 0)  {
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
 
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iterations; it++) {
   a += MK;
   b += KN;
   c += MN;
   a_p += MK;
   b_p += KN;
   c_p += MN;
   if (a == BOUND) {a = A;  b = B; c = C; }
   if (a_p == BOUND) {a_p = A;  b_p = B; c_p =C; }
  }
  l_end = libxsmm_timer_tick();
  l_total -=  libxsmm_timer_duration(l_start, l_end);

  flops = (double)64  * (double)(2 * M * K * N) * (double)iterations;
  printf("GFLOPS KERNEL = %.5g (%.4g%% efficiency) \n", (flops*1e-9)/l_total,(flops*1e-9)/l_total *100.0 / 4915.0 );
  printf("Enforce printing %lld %lld %lld %lld %lld %lld\n", a,b,c,a_p, b_p,c_p);
  shared_total = l_total;

#ifdef ERROR_CHECK
  for ( mi = 0; mi < M; mi++   ) {
    for ( mj = 0; mj < N; mj++  ) {
      for ( mk = 0; mk < K; mk++  ) {
        C_check[mj*M + mi] += A[ mk*M+mi  ] * B[  mk + mj*K   ];
      }
    }
  }

  int correct = 1;

  for ( mi = 0; mi < M; mi++   ) {
    for ( mj = 0; mj < N; mj++  ) {
      if (   fabs( C_check[mj*M+mi] - C[ mj*M+mi  ] ) > 0.0000001  ) {
         correct = 0;
      }
    }
  }

  if ( correct == 1  ) printf("Correct!!!\n");
  else  printf("Error!!!\n");
#endif


 }
}
 printf("\n\n\n");
 libxsmm_free(As);
 libxsmm_free(Bs);
 libxsmm_free(Cs);

 return 0;
}

