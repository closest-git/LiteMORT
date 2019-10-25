#pragma once
#include <omp.h>

#define OMP_FOR_func(lambda_func)											\
for (int thread = 0; thread < num_threads; thread++) {			\
	size_t start = thread*step, end = MIN2(start + step, dim), i;	\
	for (i = start; i < end; i++) {	{lambda_func;}	}		\
}

namespace Grusoft{
	inline int  OMP_FOR_STATIC_1(const size_t nSamp, size_t& step,int min_size=64, int flag = 0x0) {
		int num_threads = 1;
		step = nSamp;
		if (nSamp > min_size) {
#pragma omp parallel	
#pragma omp master											
			{	num_threads = omp_get_num_threads();	}
			step = (nSamp + num_threads - 1) / num_threads;
		}
		return num_threads;
	}
}