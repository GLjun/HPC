/*************************************************************************
    > File Name: test.c
    > Author: cgn
    > Func: 
    > Created Time: 六 12/16 20:15:16 2017
 ************************************************************************/

#include <stdio.h>
#include <omp.h>
#include <unistd.h>


void test()
{
	int i;
	printf("nested: %d\n", omp_get_nested());
	omp_set_nested(1);
	#pragma omp parallel
	{
		printf("parallel id : %d\n", omp_get_thread_num());
		#pragma omp sections
		{
			#pragma omp section 
			{
				printf("\tsection1 id : %d\n", omp_get_thread_num());
				#pragma omp parallel for schedule(static) num_threads(3)
				for(i = 0;i < 10; ++i)
				{
					printf("\t\ti : %d, id : %d\n", i, omp_get_thread_num());
				}
			}
			#pragma omp section
			{
				printf("section1 id : %d\n", omp_get_thread_num());
				int j;
				for(j = 0;j < 20; j++)
				{
					int k = 0;
					printf("\t\t\tsection1 id : %d\n", omp_get_thread_num());
				}
				sleep(4);
			}

		}
	}
}

void test2()
{
	int i;
	#pragma omp parallel num_threads(4)
	{
		#pragma omp single nowait
		{
			printf("single , id : %d\n", omp_get_thread_num());
			sleep(3);
		}
		#pragma omp for
		for(i = 0; i < 10; ++i)
		{
			printf("\tfor , i : %d, id : %d\n", i, omp_get_thread_num());
		}
	}
}

void test3()
{
	int i;
	#pragma omp parallel num_threads(4)
	{
		#pragma omp master
		{
			printf("single , id : %d\n", omp_get_thread_num());
			sleep(3);
		}
		#pragma omp for
		for(i = 0; i < 10; ++i)
		{
			printf("\tfor , i : %d, id : %d\n", i, omp_get_thread_num());
		}
	}
}

void test4()
{
	int i;
	#pragma omp parallel num_threads(1)
	{
		{
			printf("single , id : %d\n", omp_get_thread_num());
			sleep(3);
		}
	}
		#pragma omp parallel for num_threads(3)
		for(i = 0; i < 10; ++i)
		{
			printf("\tfor , i : %d, id : %d\n", i, omp_get_thread_num());
		}
}

int main()
{
	test();
	return 0;
}
