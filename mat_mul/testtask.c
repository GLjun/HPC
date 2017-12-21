/*************************************************************************
    > File Name: testtask.c
    > Author: cgn
    > Func: 
    > Created Time: 二 12/19 19:35:40 2017
 ************************************************************************/

#include <stdio.h>
#include <omp.h>
#include <unistd.h>


void testtask()
{
	#pragma omp parallel num_threads(4)
	{
#pragma omp task
		{
			printf("task 1, id : %d\n", omp_get_thread_num());
			#pragma omp task
			{
				printf("\ttask 1-1, id : %d\n", omp_get_thread_num());
			}
		}

#pragma omp task
		{
			printf("task 1, id : %d\n", omp_get_thread_num());
			#pragma omp task
			{
				printf("\ttask 1-1, id : %d\n", omp_get_thread_num());
			}
		}
	}
}

void testsingle()
{
#pragma omp master 
	{
		printf("single , id %d\n", omp_get_thread_num());
		sleep(3);
	}
#pragma omp parallel 
	{
#pragma omp single nowait
	{
		printf("single , id %d\n", omp_get_thread_num());
		sleep(3);
	}
		printf("parallel , id %d\n", omp_get_thread_num());

	}
}

void testmux()
{
	omp_lock_t lock;
	omp_init_lock(&lock);
#pragma omp parallel num_threads(2)
	{
#pragma omp sections
		{	
#pragma omp section
			{
				omp_set_lock(&lock);
				printf("set lock succ, id %d\n", omp_get_thread_num());

				omp_set_lock(&lock);
				printf("set lock succ 2, id %d\n", omp_get_thread_num());
			}
#pragma omp section
			{
				sleep(3);
				omp_unset_lock(&lock);
				printf("unset succ , id %d\n", omp_get_thread_num());
			}
		}
	}
}
int main()
{
	testmux();
	return 0;
}
