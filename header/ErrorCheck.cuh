/*
 * ErrorCheck.cuh
 *
 *  Created on: Mar 10, 2023
 *      Author: bongwon
 */

#ifndef HEADER_ERRORCHECK_CUH_
#define HEADER_ERRORCHECK_CUH_

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
    	  exit(code);
   }
}

#endif /* HEADER_ERRORCHECK_CUH_ */
