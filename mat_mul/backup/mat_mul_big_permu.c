/*************************************************************************
    > File Name: mat_mul.c
    > Author: cgn
    > Func: 
    > Created Time: 2017年12月05日 星期二 00时27分45秒
 ************************************************************************/

#include <stdio.h>
#include <omp.h>
#include "mkl.h"
#include "immintrin.h"

typedef double FT;
typedef double* PFT;

//kc mush divided by 4, for the pack of block of A
#define partition 16384
#define kc 64
#define mc 32
#define mr 4
#define nr 4

#define E(a,i,j,rows) a[(j)*(rows)+(i)]
#define I(i,j,rows) ((j)*(rows)+(i))

extern void print_matrix(PFT A, int n);
extern void init_matrix(PFT A, int n, int flag);


void block_panel_mul(FT AB[restrict], FT BP[restrict], FT C[restrict], int n, int nc, int ic)
{
	int jr, ir, i, j, si, sj, pi;
	double sum;
	__m256d ar, bc, tmp;
	PFT CB = _mm_malloc(mc*nr*sizeof(FT), 32);
	#pragma ivdep
	#pragma vector aligned
	for(jr = 0; jr < nc; jr += nr)
	{
		memset(CB, 0, mc*nr*sizeof(FT));
		for(ir = 0; ir < mc; ir += mr)
		{
			//calculate mr*nr = mr*kc *** kc*nr
			
			for(i = 0;i < mr; i++)
			{
				si = (ir + i) * kc; 
				sj = jr * kc;
				for(pi = 0; pi < nr; pi ++)
				{
					//sum = 0;
					tmp = _mm256_setzero_pd();
					for(j = 0; j < kc; j += 4){
						/*sum += AB[si + j]*BP[sj+j] +
							   AB[si + j + 1]*BP[sj+j+1] +
							   AB[si + j + 2]*BP[sj+j+2] +
							   AB[si + j + 3]*BP[sj+j+3];*/
						ar = _mm256_load_pd(&AB[si+j]);
						bc = _mm256_load_pd(&BP[sj+j]);
						tmp = _mm256_fmadd_pd(ar, bc, tmp);
							   
					}
					CB[pi*mc + ir + i] += tmp[0]+tmp[1]+tmp[2]+tmp[3];
					sj += kc;//Bj or A is in cache1
				}
			}
		}
		//update C
		for(j = 0;j < nr; j++)
		{
			for(i = 0;i < mc; i++)
			{
				E(C,ic+i, jr+j, n) = E(CB,i,j,mc);
			}
		}
	}
	_mm_free(CB);
}

void mat_mat_mul(FT A[restrict], FT B[restrict], FT C[restrict], int n)
{
	int jc, pc, ic, jr, ir;
	int i, j;
	int nc = (partition < n) ? partition : n;
	PFT pB = B;
	PFT pC = C;

	int num_threads = omp_get_num_procs();
	printf("threads : %d\n\n", num_threads);

	omp_set_num_threads(num_threads);

	for(jc = 0; jc < n; jc += nc)
	{
		pB = B + I(0, jc, n);
		pC = C + I(0, jc, n);
		for(pc = 0; pc < n; pc += kc)
		{
			//pack panel of B
			PFT BP = _mm_malloc(kc * nc * sizeof(FT) , 32);
			for(j = 0;j < nc;j ++)
			{
				memcpy(BP + j*kc, pB + kc*pc + j*n, kc * sizeof(FT)); //should be optimized to <<
			}
			
			
			#pragma omp parallel for private(i, j) schedule(dynamic)
			for(ic = 0; ic < n;ic += mc)
			{
				//pack Block of A
				PFT ABlock = _mm_malloc(mc * kc * sizeof(FT), 32);
				int left_up, ai;
				#pragma ivdep
				#pragma vector aligned
				for(i = 0;i < mc;i += mr)
				{
					ai = i*kc;
					left_up = I(ic+i, pc, n);
					for(j = 0; j < kc; j += 4)
					{
						ABlock[ai] = A[left_up];
						ABlock[ai+kc] = A[left_up+1];
						ABlock[ai+kc*2] = A[left_up+2];
						ABlock[ai+kc*3] = A[left_up+3];
						left_up += n;
						
						ABlock[ai+1] = A[left_up];
						ABlock[ai+1+kc] = A[left_up+1];
						ABlock[ai+1+kc*2] = A[left_up+2];
						ABlock[ai+1+kc*3] = A[left_up+3];
						left_up += n;
	
						ABlock[ai+2] = A[left_up];
						ABlock[ai+2+kc] = A[left_up+1];
						ABlock[ai+2+kc*2] = A[left_up+2];
						ABlock[ai+2+kc*3] = A[left_up+3];
						left_up += n;
	
						ABlock[ai+3] = A[left_up];
						ABlock[ai+3+kc] = A[left_up+1];
						ABlock[ai+3+kc*2] = A[left_up+2];
						ABlock[ai+3+kc*3] = A[left_up+3];
						left_up += n;
	
						ai += 4;
					}
				}
				block_panel_mul(ABlock, BP, pC, n,nc, ic);
	
	
				_mm_free(ABlock);
				
				/*for(jr = 0; jr < kc; jr += nr)
				{
					for(ir = 0; ir < mc; ir += mr)
					{
						
					}
				}
				_mm_free(BP);	*/
			}
			
			_mm_free(BP);
		}
	
	}


}

int main()
{
	int n = 8192;
	int i;
	int loopcnt = 1;
	PFT A = _mm_malloc(n*n*sizeof(FT), 32);
	PFT B = _mm_malloc(n*n*sizeof(FT), 32);
	PFT C = _mm_malloc(n*n*sizeof(FT), 32);

	init_matrix(A, n, 1);
	init_matrix(B, n, 1);
	init_matrix(C, n, 0);

	for(i = 0;i < loopcnt; i++)
	{
		double tc = dsecnd();
		mat_mat_mul(A,B,C,n);
		double tc2 = dsecnd();
		printf("time : %.5f milliseconds \n\n", ((tc2-tc))*1000);
	}

	/*printf("========A========\n");
	print_matrix(A, n);
	printf("========B========\n");
	print_matrix(B, n);
	printf("========C========\n");
	print_matrix(C, n);*/

	_mm_free(A);
	_mm_free(B);
	_mm_free(C);

	return 0;
}

void print_matrix(PFT A, int n)
{
	int i, j;
	for(i = 0;i < n;i ++)
	{
		for(j = 0;j < n;j ++)
		{
			printf("%.5f ", E(A,i,j,n));
		}
		printf("\n");
	}
}

void init_matrix(PFT A, int n, int flag)
{
	int i, j;
	if(!flag)
	{
		for(i = 0;i < n;i ++)
		{
			for(j = 0; j < n; j ++)
			{
				E(A,i,j,n) = 0;
			}
		}
		
	}
	else
	{
		for(i = 0;i < n;i ++)
		{
			for(j = 0; j < n; j ++)
			{
				E(A,i,j,n) = fmod(i*j*1.0, 10.0) + 1;
				//E(A,i,j,n) = 1.0;
			}
		}
	}
}
