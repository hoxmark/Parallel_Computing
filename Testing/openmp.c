// Compile with:
// gcc -std=c99 openmp.c -fopenmp

#include <stdio.h>
#include <omp.h>

int main(){

#pragma omp parallel for num_threads(3)
	
	    for(int c = 0; c < 10; c++){
	        printf("Bear Iteration: %d, Thread: %d of %d\n", c, omp_get_thread_num(), omp_get_num_threads());

	    }
	
}
