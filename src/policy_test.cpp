/******************************************************************************
* FILE: omp_hello.c
* DESCRIPTION:
*   OpenMP Example - Hello World - C/C++ Version
*   In this simple example, the master thread forks a parallel region.
*   All threads in the team obtain their unique thread number and print it.
*   The master thread only prints the total number of threads.  Two OpenMP
*   library routines are used to obtain the number of threads and each
*   thread's number.
* AUTHOR: Blaise Barney  5/99
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>
#include <iostream>

int main (int argc, char *argv[]) {
    int nthreads;  

    int final_nthreads_1 = -1;
    int final_nthreads_2 = -1;

    /* Fork a team of threads giving them their own copies of variables */
    for(int i = 0; i < 100; ++i) {
#pragma omp parallel private(nthreads)
        {
            nthreads = omp_get_num_threads();
            struct timespec tim, tim2;
            tim.tv_sec = 0;
            long diff = abs(4 - nthreads);
            tim.tv_nsec = (diff * 2000000) + 1000000;
            nanosleep(&tim, &tim2);
            if(i == 99) {
                #pragma omp master
                {
                    final_nthreads_1 = nthreads;   
                }
            }
        }


#pragma omp parallel private(nthreads)
        {
            nthreads = omp_get_num_threads();
            struct timespec tim, tim2;
            tim.tv_sec = 0;
            long diff = abs(8 - nthreads);
            tim.tv_nsec = (diff * 2000000) + 1000000;
            nanosleep(&tim, &tim2);
            if(i == 99) {
                #pragma omp master
                {
                    final_nthreads_2 = nthreads;   
                }
            }
        }
    }

    std::cerr << std::endl;
    std::cerr << "Final omp_num_threads for region 1: " << final_nthreads_1 << " (should be 4)" << std::endl;
    std::cerr << "Final omp_num_threads for region 2: " << final_nthreads_2 << " (should be 8)" << std::endl;
    if(final_nthreads_1 == 4 && final_nthreads_2 == 8) {
        std::cerr << "Test passed." << std::endl;
    } else {
        std::cerr << "Test failed." << std::endl;
    }
    std::cerr << std::endl;


}

