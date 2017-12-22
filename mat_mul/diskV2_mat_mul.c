/*************************************************************************
    > File Name: mat_mat_mul.c
    > Author: cgn
    > Func: matrix multiplication V3
    > Created Time: 五 12/15 11:26:31 2017
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "mkl.h"
#include "immintrin.h"

typedef double FT;
typedef double* PFT;

#define DEBUG

#define CACHE_FILE_A "cacheA"
#define CACHE_FILE_B "cacheB"
#define CACHE_FILE_C "cacheC"
#define DOUBLE_MOD (1e+300)
#define THREAD_NUMS 4
#define MC 384 //256 + 128
#define KC 384 //256 + 128
#define NC 4096
#define MEMORY_MAX_SIZE 2147483648LL //2G
#define BC_PANEL_MAX_SIZE 536870912LL //256M < MEMORY_MAX_SIZE/6

#define MR 4
#define NR 4

#define I(i,j,rows) ((i)+(j)*rows)

static FT BP[KC * NC] __attribute__ ((aligned (32)));
static FT ABArray[MC * KC * THREAD_NUMS] __attribute__ ((aligned (32)));

void BPanel_kcnc(FT B[restrict], const int kc, const int nc, FT BP[restrict], const int n)
{
	int i, j, jr;
	int nb = nc/NR, rn = nc % NR;
	for(jr = 0; jr < nb; ++jr)
	{
		for(i = 0; i < kc; ++i)
		{
			for(j = 0; j < NR; ++j){
				BP[j] = B[j*n];
			}
			BP += NR;
			++ B;
		}
		B += NR * n - kc; // minus kc to the top row of BP
	}

	if(rn)
	{
		for(i = 0; i < kc; ++i)
		{
			for(j = 0; j < rn; ++j) 
			{
				BP[j] = B[j*n];
			}
			for(j = rn; j < NR; ++j)
			{
				BP[j] = 0.0;
			}
			BP += NR;
			++ B;
		}
	}

}

void ABlock_mcnc(FT A[restrict], const int mc, const int kc, FT AB[restrict], const int n)
{
	int i, j, kr;
	int mb = mc/MR, rm = mc % MR;

	for(kr = 0; kr < mb; ++ kr)
	{
		for(j = 0; j < kc; ++j)
		{
			for(i = 0; i < MR; ++i)
			{
				AB[i] = A[i];
			}
			AB += MR;
			A += n;
		}
		A += MR - kc*n; // minus kc*n
	}

	if(rm)
	{
		for(j = 0; j < kc; ++j)
		{
			for(i = 0; i < rm; ++i)
			{
				AB[i] = A[i];
			}
			for(i = rm; i < MR; ++i)
			{
				AB[i] = 0;
			}
			AB += MR;
			A += n;
		}
	}

}

void mat_v_v(const FT AV[restrict], const FT BV[restrict], FT C[restrict], const int kc, const int n)
{

	int i;
	//FT TMPC[MR*NR] __attribute__ ((aligned (32)));
	
	register __m256d t0, t1, t2, t3;
	register __m256d a;
	register __m256d b0, b1, b2, b3;

	t0 = _mm256_setzero_pd();
	t1 = _mm256_setzero_pd();
	t2 = _mm256_setzero_pd();
	t3 = _mm256_setzero_pd();
	for(i = 0;i < kc; ++i)
	{
		a = _mm256_load_pd(AV);
		AV += 4;
		b0 = _mm256_broadcast_sd(BV);
		t0 = _mm256_fmadd_pd(a,b0,t0);

		b1 = _mm256_broadcast_sd(BV+1);
		t1 = _mm256_fmadd_pd(a,b1,t1);

		b2 = _mm256_broadcast_sd(BV+2);
		t2 = _mm256_fmadd_pd(a,b2,t2);

		b3 = _mm256_broadcast_sd(BV+3);
		t3 = _mm256_fmadd_pd(a,b3,t3);
		BV += 4;
	}

	b0 = _mm256_load_pd(C);
	b1 = _mm256_load_pd(C+n);
	t0 = _mm256_add_pd(b0, t0);
	_mm256_store_pd(C, t0);
	b2 = _mm256_load_pd(C+(n<<1));
	t1 = _mm256_add_pd(b1, t1);
	_mm256_store_pd(C+n, t1);
	b3 = _mm256_load_pd(C+n*3);
	t2 = _mm256_add_pd(b2, t2);
	_mm256_store_pd(C+(n<<1), t2);
	t3 = _mm256_add_pd(b3, t3);
	_mm256_store_pd(C+n*3, t3);

	/*_mm256_store_pd(TMPC, t0);
	_mm256_store_pd(TMPC+4, t1);
	_mm256_store_pd(TMPC+8, t2);
	_mm256_store_pd(TMPC+12, t3);*/


}

void mat_block_panel(const FT AB[restrict], const FT BP[restrict], FT C[restrict],
		const int mc, const int kc, const int nc, const int n)
{
	int ir, jr, k, mr, nr;
	int nb = (nc+NR-1)/NR, rn = nc % NR;
	int mb = (mc+MR-1)/MR, rm = mc % MR;
	FT TMPC[MR*NR] __attribute__ ((aligned (32)));
	

	for(jr = 0; jr < nb; ++jr)
	{
		nr = (jr != nb-1 || !rn) ? NR : rn;
		for(ir = 0; ir < mb; ++ir)
		{
			mr = (ir != mb-1 || !rm) ? MR : rm;
			if(MR == mr && NR == nr)
			{
				mat_v_v(&AB[ir*kc*MR], &BP[jr*kc*NR], &C[I(ir*MR, jr*NR, n)], kc, n);
			}
			else
			{
				mat_v_v(&AB[ir*kc*MR], &BP[jr*kc*NR], TMPC, kc, MR);
				//update tail of c
				PFT p = &C[I(ir*MR, jr*NR, n)];
				int i,j;
				for(j = 0; j < nr; ++j)
				{
					for(i = 0; i < mr; ++i)
					{
						p[i + j*n] += TMPC[i + j*MR];
					}
				}
			}
		}
	}

}


void mat_mat_mul(FT A[restrict], FT B[restrict], FT C[restrict], const int n)
{
	
	int threads = omp_get_num_procs();
	int thread_id = 0;
	int ic, jc, lc, nc, mc, kc;
	int nb = (n+NC-1)/NC, rn = n % NC;
	int mb = (n+MC-1)/MC, rm = n % MC;
	int kb = (n+KC-1)/KC, rk = n % KC;
	printf("threads %d\n\n", threads);



	for(jc = 0; jc < nb; ++jc)
	{
		nc = (jc != nb-1 || !rn) ? NC : rn;
		for(lc = 0; lc < kb; ++lc)
		{
			kc = (lc != kb-1 || !rk) ? KC : rk;
			//pack B into panels
			BPanel_kcnc(&B[I(lc*KC, jc*NC, n)], kc, nc, BP, n);

#pragma omp parallel for private(ic) num_threads(threads) 
			for(ic = 0;ic < mb; ++ic)
			{
				thread_id = omp_get_thread_num();
				mc = (ic != mb-1 || !rm) ? MC : rm;
				ABlock_mcnc(&A[I(ic*MC, lc*KC, n)], mc, kc, &ABArray[thread_id * MC * KC], n);
				mat_block_panel(&ABArray[thread_id * MC * KC], BP, &C[I(ic*MC, jc*NC, n)], mc, kc, nc, n);
			}
		}
	}

}

void read_matrix(FT M[restrict], int n, int m, FILE* f)
{
	fread((void*)M, sizeof(char), (long)n*(long)m*sizeof(FT), f);
}

void write_matrix(FT M[restrict], int n, int m, FILE* f)
{
	fwrite((void*)M, sizeof(char), (long)n*(long)m*sizeof(FT), f);
}

void dynamic_mat_mul(FT A[restrict], FT B[restrict], FT C[restrict], int n, int wbc, int WA, int offset_A, FILE* fa)
{
	int thread_id = 0;
	int ic, jc, lc, nc, mc, kc;
	int nb = (wbc+NC-1)/NC, rn = wbc % NC;
	int mb = (n + MC-1)/MC, rm = n   % MC;
	int kb = (n + KC-1)/KC, rk = n   % KC;
	int offset = offset_A;

	for(jc = 0; jc < nb; ++jc)
	{
		nc = (jc != nb-1 || !rn) ? NC : rn;
		for(lc = 0; lc < kb; ++lc)
		{
			kc = (lc != kb-1 || !rk) ? KC : rk;
			BPanel_kcnc(&B[I(lc*KC, jc*NC, n)], kc, nc, BP, n);
			#pragma omp parallel num_threads(THREAD_NUMS)
			{
				#pragma omp master //read next KC panel of A
				{
					if(lc == kb-1 && jc != nb-1)
						fseek(fa, 0, SEEK_SET);
					if(jc != nb-1 || lc != kb-1)
						read_matrix(&A[KC*n*offset], n, ((lc != kb-2 || !rk) ? KC : rk), fa);
				}
				#pragma omp for
				for(ic = 0; ic < mb; ++ic)
				{
					thread_id = omp_get_thread_num();
					mc = (ic != mb-1 || !rm) ? MC : rm;
					ABlock_mcnc(&A[I(ic*MC, KC*(1-offset), n)], mc, kc, &ABArray[thread_id * MC * KC], n);
					mat_block_panel(&ABArray[thread_id * MC * KC], BP, &C[I(ic*MC, jc*NC, n)], mc, kc, nc, n);
				}

			}	
			
			offset = 1 - offset;

		}
	}
}


void print_matrix(PFT A, int n)
{
	int i, j;
	for(i = 0;i < n;i ++)
	{
		for(j = 0;j < n;j ++)
		{
			printf("%.5f ", A[I(i,j,n)]);
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
				A[I(i,j,n)] = 0.0;
			}
		}
		
	}
	else
	{
		for(i = 0;i < n;i ++)
		{
			for(j = 0; j < n; j ++)
			{
				A[I(i,j,n)] = fmod(i*j*1.0, 10.0) + 1;
			}
		}
	}
}

double sumMatirx(FT M[restrict], int m, int n)
{
	int i, j;
	double sum = 0;
	for(j = 0; j < n; ++j)
	{
		for(i = 0; i < m; ++i)
		{
			sum = (sum < DOUBLE_MOD-M[I(i,j,m)]) ? sum+M[I(i,j,m)] : sum+M[I(i,j,m)] - DOUBLE_MOD;
		}
	}

	return sum;

}

void process_in_memory(int n)
{
	int i;
	int loopcnt = 1;
	double tc, tc2;

	PFT A = _mm_malloc(n*n*sizeof(FT), 32);
	PFT B = _mm_malloc(n*n*sizeof(FT), 32);
	PFT C = _mm_malloc(n*n*sizeof(FT), 32);

	init_matrix(A, n, 1);
	init_matrix(B, n, 1);
	init_matrix(C, n, 0);


	for(i = 0;i < loopcnt; i++)
	{
		tc = dsecnd();
		mat_mat_mul(A, B, C, n);
		tc2 = dsecnd();
		printf("mmt time : %.5f milliseconds \n\n", ((tc2-tc))*1000.0);
	}

	double checkSum1 = sumMatirx(C, n, 10);
	
	init_matrix(C, n, 0);

	for(i = 0;i < loopcnt; i++)
	{
		tc = dsecnd();
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n,n,n, 1.0, A, n, B, n, 0.0, C, n);
		tc2 = dsecnd();
		printf("cblas time : %.5f milliseconds \n\n", ((tc2-tc))*1000.0);
	}
	double checkSum2 = sumMatirx(C, n, 10);
	printf("mmtsum : %.5f, cblassum : %.5f, diff : %.5f\n\n", checkSum1, checkSum2, (checkSum1-checkSum2));

	//print_matrix(A, n);
	//print_matrix(B, n);
	//print_matrix(C, n);

	_mm_free(A);
	_mm_free(B);
	_mm_free(C);

}

void init_matrix_rectangle(FT M[restrict], int m, int n, int flag)
{
	double random_seed = 0.0;
	long long j;
	long long size = (long)m * (long)n;
	if(flag)
	{
		#pragma vector aligned
		for(j = 0; j < size; ++j)
		{
			M[j] = 1.0 + random_seed;
		}
	}
	else
	{
		#pragma vector aligned
		for(j = 0; j < size; ++j)
		{
			M[j] = 0.0;
		}
	}
}

void init_matrix_disk(FT A[restrict], FT B[restrict], FT C[restrict], FILE* a, FILE* b, int n, int WBC, int WA)
{
	#pragma omp parallel sections num_threads(3)
	{
		#pragma omp section
		{
			int i, j, wc;
			int wb = (n+WA-1)/WA, rw = n % WA;
			for(j = 0; j < wb; ++j)
			{
				wc = (j != wb - 1 || !rw) ? WA : rw;
				init_matrix_rectangle(A, n, wc, 1);
				fwrite((void*)A, sizeof(char), (long)n*(long)wc*sizeof(FT), a);
			}
			fflush(a);
		}
		#pragma omp section
		{
			int i, j, wc;
			int wb = (n+WBC-1)/WBC, rw = n % WBC;
			for(j = 0; j < wb; ++j)
			{
				wc = (j != wb - 1 || !rw) ? WBC : rw;
				init_matrix_rectangle(B, n, wc, 1);
				fwrite((void*)B, sizeof(char), (long)n*(long)wc*sizeof(FT), b);
			}
			fflush(b);
		}
		#pragma omp section
		{
			init_matrix_rectangle(C, n, WBC, 0);
		}
	}

#ifdef DEBUG
	printf(">> Finish initializing matrix A, B, C\n");
#endif

}

void process_out_of_memory(int n)
{

	int iw, wbc, offset_A, offset_B, offset_C;

	int max_w = BC_PANEL_MAX_SIZE / 8 / n;
	int WBC = (max_w > NC) ? (max_w/NC)*NC : max_w;// width of panel of B/C
	int wb = (n+WBC-1)/WBC, rw = n % WBC;
#ifdef DEBUG
	printf("max_w:%d, WBC:%d, wb:%d\n", max_w, WBC, wb);
#endif

	//int WA = (WBC/KC)*KC; //WA should be divided by KC
	int WA = 2*KC;// for simple


	PFT A = _mm_malloc(2*WA*n*sizeof(FT), 32);
	PFT B = _mm_malloc(2*WBC*n*sizeof(FT), 32);
	PFT C = _mm_malloc(2*WBC*n*sizeof(FT), 32);
	offset_A = 1;
	offset_B = 1;
	offset_C = 1;
	//init matrix
	FILE* fa = fopen(CACHE_FILE_A, "wb");
	FILE* fb = fopen(CACHE_FILE_B, "wb");
	FILE* fc = fopen(CACHE_FILE_C, "wb");
	init_matrix_disk(A, B, C, fa, fb, n, WBC, WA);

	//close fa, fb
	fclose(fa);
	fclose(fb);
	//reopen with mode read 
	fa = fopen(CACHE_FILE_A, "rb");
	fb = fopen(CACHE_FILE_B, "rb");

	//open nested thread
	omp_set_nested(1);
	for(iw = 0;iw < wb; ++iw)
	{
		wbc = (iw != wb-1 || !rw) ? WBC : rw;
#ifdef DEBUG
		printf("iw/wb:%d/%d, wbc:%d, offset_A:%d, offset_B:%d, offset_C:%d\n", iw, wb, wbc, offset_A, offset_B, offset_C);
#endif
		#pragma omp parallel num_threads(3)
		{
			#pragma omp master // read next Panel of B
			{
				if(iw < wb-1) // wb = 1
				{
					read_matrix(&B[WBC*n*offset_B], n, ((iw != wb-2 || !rw) ? WBC : rw), fb);
				}
			}
			#pragma omp single nowait // write panel of C
			{
				if(iw > 0)
				{
					write_matrix(&C[WBC*n*offset_C], n, WBC, fc);
					memset((void*)(&C[WBC*n*offset_C]), 0, WBC*n*sizeof(FT));
				}
			}
			#pragma omp single // computing
			{
				fseek(fa, 0, SEEK_SET);
				read_matrix(A, n, KC, fa);
				dynamic_mat_mul(A, &B[WBC*n*(1-offset_B)], &C[WBC*n*(1-offset_C)], n, wbc, WA, offset_A, fa);
			}
		}
		offset_B = 1 - offset_B;
		if(iw > 0)
			offset_C = 1 - offset_C;
	}
#ifdef DEBUG
	printf(">> Finish computing!!!\n");
	printf(">> Begin writing last panel of C\n");
#endif

	//write last panle of matrix
	write_matrix(&C[WBC*n*offset_C], n, (rw != 0) ? rw : WBC, fc);
}

int main()
{

	int n = 8192;

	process_out_of_memory(n);

	return 0;
}
