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


void block_panel_mul(PFT AB, PFT BP, PFT C, int n, int nc, int ic)
{
	int jr, ir, i, j, si, sj, pi;
	double sum;
	double *p, *pBC;
	__m256d ze = _mm256_setzero_pd();
	PFT CB = _mm_malloc(mc*nr*sizeof(FT), 32);
	for(jr = 0; jr < nc; jr += nr)
	{
		memset(CB, 0, mc*nr*sizeof(FT));
		for(ir = 0; ir < mc; ir += mr)
		{
			//calculate mr*nr = mr*kc *** kc*nr
			
			for(i = 0;i < mr; i++)
			{
				si = ir * kc; // A 按照 mr*kc 块存储的
				sj = jr * kc;
				for(pi = 0; pi < nr; pi ++)
				{
					/*sum = 0;
					for(j = 0; j < kc; j += 4){
						sum += AB[si + j*mr + i]*BP[sj+j] +
							   AB[si + (j+1)*mr + i]*BP[sj+j+1] +
							   AB[si + (j+2)*mr + i]*BP[sj+j+2] +
							   AB[si + (j+3)*mr + i]*BP[sj+j+3];
							   
					}
					CB[pi*mc + ir + i] += sum;*/
					p = &AB[si + i];
					pBC = &BP[sj];
					__m256d ar0 = _mm256_set_pd(*(p+3*mr), *(p+2*mr), *(p+mr), *(p));
					__m256d bc0 = _mm256_load_pd(pBC);
					__m256d sum0 = _mm256_fmadd_pd(ar0, bc0, ze);

					__m256d ar1 = _mm256_set_pd(*(p+7*mr), *(p+6*mr), *(p+5*mr), *(p+4*mr));
					__m256d bc1 = _mm256_load_pd(pBC+4);
					sum0 = _mm256_fmadd_pd(ar1, bc1, sum0);
					
					__m256d ar2 = _mm256_set_pd(*(p+11*mr), *(p+10*mr), *(p+9*mr), *(p+8*mr));
					__m256d bc2 = _mm256_load_pd(pBC+8);
					sum0 = _mm256_fmadd_pd(ar2, bc2, sum0);

					__m256d ar3 = _mm256_set_pd(*(p+15*mr), *(p+14*mr), *(p+13*mr), *(p+12*mr));
					__m256d bc3 = _mm256_load_pd(pBC+12);
					sum0 = _mm256_fmadd_pd(ar3, bc3, sum0);

					__m256d ar4 = _mm256_set_pd(*(p+19*mr), *(p+18*mr), *(p+17*mr), *(p+16*mr));
					__m256d bc4 = _mm256_load_pd(pBC+16);
					sum0 = _mm256_fmadd_pd(ar4, bc4, sum0);

					__m256d ar5 = _mm256_set_pd(*(p+23*mr), *(p+22*mr), *(p+21*mr), *(p+20*mr));
					__m256d bc5 = _mm256_load_pd(pBC+20);
					sum0 = _mm256_fmadd_pd(ar5, bc5, sum0);

					__m256d ar6 = _mm256_set_pd(*(p+27*mr), *(p+26*mr), *(p+25*mr), *(p+24*mr));
					__m256d bc6 = _mm256_load_pd(pBC+24);
					sum0 = _mm256_fmadd_pd(ar6, bc6, sum0);

					__m256d ar7 = _mm256_set_pd(*(p+31*mr), *(p+30*mr), *(p+29*mr), *(p+28*mr));
					__m256d bc7 = _mm256_load_pd(pBC+28);
					sum0 = _mm256_fmadd_pd(ar7, bc7, sum0);

					__m256d ar8 = _mm256_set_pd(*(p+35*mr), *(p+34*mr), *(p+33*mr), *(p+32*mr));
					__m256d bc8 = _mm256_load_pd(pBC+32);
					sum0 = _mm256_fmadd_pd(ar8, bc8, sum0);

					__m256d ar9 = _mm256_set_pd(*(p+39*mr), *(p+38*mr), *(p+37*mr), *(p+36*mr));
					__m256d bc9 = _mm256_load_pd(pBC+36);
					sum0 = _mm256_fmadd_pd(ar9, bc9, sum0);

					__m256d ar10 = _mm256_set_pd(*(p+43*mr), *(p+42*mr), *(p+41*mr), *(p+40*mr));
					__m256d bc10 = _mm256_load_pd(pBC+40);
					sum0 = _mm256_fmadd_pd(ar10, bc10, sum0);

					__m256d ar11 = _mm256_set_pd(*(p+47*mr), *(p+46*mr), *(p+45*mr), *(p+44*mr));
					__m256d bc11 = _mm256_load_pd(pBC+44);
					sum0 = _mm256_fmadd_pd(ar11, bc11, sum0);

					__m256d ar12 = _mm256_set_pd(*(p+51*mr), *(p+50*mr), *(p+49*mr), *(p+48*mr));
					__m256d bc12 = _mm256_load_pd(pBC+48);
					sum0 = _mm256_fmadd_pd(ar12, bc12, sum0);

					__m256d ar13 = _mm256_set_pd(*(p+55*mr), *(p+54*mr), *(p+53*mr), *(p+52*mr));
					__m256d bc13 = _mm256_load_pd(pBC+52);
					sum0 = _mm256_fmadd_pd(ar13, bc13, sum0);

					__m256d ar14 = _mm256_set_pd(*(p+59*mr), *(p+58*mr), *(p+57*mr), *(p+56*mr));
					__m256d bc14 = _mm256_load_pd(pBC+56);
					sum0 = _mm256_fmadd_pd(ar14, bc14, sum0);
					
					__m256d ar15 = _mm256_set_pd(*(p+63*mr), *(p+62*mr), *(p+61*mr), *(p+60*mr));
					__m256d bc15 = _mm256_load_pd(pBC+60);
					sum0 = _mm256_fmadd_pd(ar15, bc15, sum0);

					
					CB[pi*mc + ir + i] += sum0[0] + sum0[1] + sum0[2] + sum0[3];
					
					

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
