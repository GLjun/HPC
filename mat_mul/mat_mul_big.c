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
#define kc 256
#define mc 256
#define mr 4
#define nr 4

#define E(a,i,j,rows) a[(j)*(rows)+(i)]
#define I(i,j,rows) ((j)*(rows)+(i))

extern void print_matrix(PFT A, int n);
extern void init_matrix(PFT A, int n, int flag);


void block_panel_mul(PFT AB, PFT BP, PFT C, int n, int nc, int ic)
{
	int jr, ir, i, j, si, sj, pi, ai, bi;
	double sum;
	//__m256d ar, bc, tmp;
	//PFT CB = _mm_malloc(mc*nr*sizeof(FT), 32);
	//FT CB[mc*nr];
	FT CBR[mr*nr];
	#pragma ivdep
	#pragma vector aligned
	for(jr = 0; jr < nc; jr += nr)
	{
		//memset(CB, 0, mc*nr*sizeof(FT));
		for(ir = 0; ir < mc; ir += mr)
		{
			//calculate mr*nr = mr*kc *** kc*nr
			for(i = 0;i < mr*nr; i++)
				CBR[i] = 0;
			
			for(i = 0;i < mr; i++)
			{
				si = ir * kc; // A 按照 mr*kc 块存储的
				sj = jr * kc;
				for(pi = 0; pi < nr; pi ++)
				{
					sum = 0;
					ai = si+i;
					bi = sj;
					for(j = 0; j < kc; j += 4){
						/*sum += AB[si + j*mr + i]*BP[sj+j] +
							   AB[si + (j+1)*mr + i]*BP[sj+j+1] +
							   AB[si + (j+2)*mr + i]*BP[sj+j+2] +
							   AB[si + (j+3)*mr + i]*BP[sj+j+3];*/
						sum += AB[ai]*BP[bi] +
							   AB[ai+mr]*BP[bi+1] +
							   AB[ai+mr+mr]*BP[bi+2] +
							   AB[ai+mr+mr]*BP[bi+3];
/*						ar = _mm256_set_pd(AB[ai+mr+mr+mr], AB[ai+(mr << 1)], AB[ai+mr], AB[ai] );
						bc = _mm256_load_pd(BP+bi);
						//tmp = _mm256_mul_pd(_mm256_set_pd(AB[ai+mr+mr+mr], AB[ai+(mr<<1)], AB[ai+mr], AB[ai]), _mm256_load_pd(BP+bi));	
						tmp = _mm256_mul_pd(ar,bc);
						sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
						*/
						//sum += AB[ai]*BP[bi] + AB[ai+mr]*BP[bi+1] + AB[ai+(mr<<1)]*BP[bi+2] + AB[ai+mr +(mr<<1)]*BP[bi+3];
						ai += (mr << 2);
						bi += 4;
							   
					}
					CBR[pi*mr + i] += sum;
					sj += kc;//Bj or A is in cache1
				}
			}
			for(j = 0;j < nr; j++)
			{
				for(i = 0;i < mr; i++)
				{
					E(C, ic+i, jr+j, n) = CBR[i+j*mr];
				}
			}
		}
		//update C
		/*
		for(j = 0;j < nr; j++)
		{
			for(i = 0;i < mc; i++)
			{
				E(C,ic+i, jr+j, n) = E(CB,i,j,mc);
			}
		}*/
	}
	//_mm_free(CB);
}

void mat_mat_mul(PFT A, PFT B, PFT C, int n)
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
			
			
			//#pragma omp parallel for private(i, j) schedule(dynamic)
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
						ABlock[ai+1] = A[left_up+1];
						ABlock[ai+2] = A[left_up+2];
						ABlock[ai+3] = A[left_up+3];
						left_up += n;
						
						ABlock[ai+4] = A[left_up];
						ABlock[ai+5] = A[left_up+1];
						ABlock[ai+6] = A[left_up+2];
						ABlock[ai+7] = A[left_up+3];
						left_up += n;
	
						ABlock[ai+8] = A[left_up];
						ABlock[ai+9] = A[left_up+1];
						ABlock[ai+10] = A[left_up+2];
						ABlock[ai+11] = A[left_up+3];
						left_up += n;
	
						ABlock[ai+12] = A[left_up];
						ABlock[ai+13] = A[left_up+1];
						ABlock[ai+14] = A[left_up+2];
						ABlock[ai+15] = A[left_up+3];
						left_up += n;
	
						ai += 4*mr;
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
	int n = 1024;
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
