/*************************************************************************
    > File Name: mat_mat_mul.c
    > Author: cgn
    > Func: matrix multiplication V4
    > Created Time: 五 12/15 11:26:31 2017
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "mkl.h"
#include "immintrin.h"

typedef double FT;
typedef double* PFT;

#define PRINT_DEBUG

#define MALLOC_MAX_SIZE 2147483648LL
#define PERFORMANCE (300LL*1073741824LL) // 300GFLOPS
#define READ_SPEED 1073741824LL // 1G Bytes
#define WRITE_SPEED 2147483648LL // 2G Bytes
#define MEMORY_SIZE 1073741824LL // 1G
//#define MEMORY_SIZE 2147483648LL //2G
#define SWELL_FACTOR 1
#define WRITE_BLOCK_MAX 2147483647LL
#define READ_BLOCK_MAX 1073741824LL

#define DOUBLE_MOD (1e+300)
#define THREAD_NUMS 24
#define MC 384 //256 + 128
#define KC 384 //256 + 128
//#define NC 4096
#define NC 1024 // DEBUG

#define MR 4
#define NR 4

#define I(i,j,rows) ((i)+(j)*rows)

typedef struct FileCache {
	FILE* fhA;
	FILE* fhB;
	FILE* fhC;
} FileCache;

typedef struct MatrixDisk {
	PFT A;
	PFT B;
	PFT C;
	PFT A2;
	PFT B2;
	PFT C2;
} MatrixDisk;

static FileCache mat_file_cache = {
	NULL, NULL, NULL
};
/*static MatrixDisk mat_disk = {
	NULL, NULL, NULL, NULL, NULL, NULL
};*/
MatrixDisk mat_disk;
static FT BP[KC * NC] __attribute__ ((aligned (32)));
static FT ABArray[MC * KC * THREAD_NUMS] __attribute__ ((aligned (32)));
const char* CACHE_FILE_NAME_A = "cacheA";
const char* CACHE_FILE_NAME_B = "cacheB";
const char* CACHE_FILE_NAME_C = "cacheC";
static omp_lock_t read_lock;
static omp_lock_t change_lock;
static int READ_A_PANEL_INDEX = 0;

void BPanel_kcnc(FT B[restrict], const int kc, const int nc, FT BP[restrict], const int n)
{
	int i, j, jr;
	int nb = nc/NR, rn = nc % NR;
	for(jr = 0; jr < nb; ++jr)
	{
		for(i = 0; i < kc; ++i)
		{
#pragma vector aligned
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
#pragma vector aligned
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


void mat_mat_mul_disk(MatrixDisk* m,  const int n, const int WBC, const int WA)
{
	int threads = omp_get_num_procs() - 1;
	int thread_id = 0;
	int ic, jc, lc, nc, mc, kc, wac, wc;
	int nb = (WBC+NC-1)/NC, rn = WBC % NC;
	int mb = (n+MC-1)/MC, rm = n % MC;
	int kb, rk;
	int wab = (n+WA-1)/WA, rwa = n % WA; // WA must be divided by KC
	PFT tmp;
	printf("threads %d\n\n", threads);

	for(jc = 0; jc < nb; ++jc)
	{
		nc = (jc != nb-1 || !rn) ? NC : rn;
		for(wac = 0; wac < wab; ++ wac)
		{
			if(wac != wab-1 || !rwa)
			{
				wc = WA;
				kb = WA/KC;
				rk = 0;
			}
			else
			{
				wc = rwa;
				kb = (rwa+KC-1)/KC;
				rk = rwa % KC;
			}
			//get next WA PANEL
			//if(wac)
			{
				omp_set_lock(&change_lock);		

				//if(READ_A_PANEL == wac)
				//{
					tmp = m->A;
					m->A = m->A2;
					m->A2 = tmp;
					if(!jc && !wac) //Since the first time "wac == 0" is included, so we need to exchange B/B2, C/C2 at the beginning
					{
						tmp = m->B;
						m->B = m->B2;
						m->B2 = tmp;
						tmp = m->C;
						m->C = m->C2;
						m->C2 = tmp;
					}
					//READ_A_PANEL_INDEX = (READ_A_PANEL_INDEX < wab-1) ? (READ_A_PANEL_INDEX+1) : 0;
					READ_A_PANEL_INDEX = (wac != wab-1) ? (wac+1) : 0;
				//}
				omp_unset_lock(&read_lock);
				omp_unset_lock(&change_lock);
			}

			for(lc = 0; lc < kb; ++lc)
			{
				kc = (!rk || lc != kb-1) ? KC : rk;
				//pack B into panels
				BPanel_kcnc(&(m->B)[I(lc*KC, jc*NC, n)], kc, nc, BP, n);

#pragma	omp parallel for private(ic) num_threads(threads) 
				for(ic = 0;ic < mb; ++ic)
				{
					thread_id = omp_get_thread_num();
					mc = (ic != mb-1 || !rm) ? MC : rm;
					ABlock_mcnc(&(m->A)[I(ic*MC, lc*KC, n)], mc, kc, &ABArray[thread_id * MC * KC], n);
					mat_block_panel(&ABArray[thread_id * MC * KC], BP, &(m->C)[I(ic*MC, jc*NC, n)], mc, kc, nc, n);
				}
			}
		}
	}
	
#ifdef PRINT_DEBUG
	printf("finish calculating one iteration, n:%d, WBC:%d, WA:%d\n", n, WBC, WA);
#endif

}

void print_matrix(FT A[restrict], int n)
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
void init_matrix_offset(FT A[restrict], int m, int n, int si, int sj, int flag)
{
	int i, j;
	if(!flag)
	{
		for(j = 0; j < n; ++j)
		{
			#pragma vector aligned
			for(i = 0; i < m; ++i)
			{
				A[I(i,j,m)] = 0.0;
			}
		}
	}
	else
	{
		for(j = 0; j < n; ++j)
		{
			#pragma vector aligned
			for(i = 0; i < m; ++i)
			{
				A[I(i,j,m)] = fmod((i+si)*(j+sj)*1.0, 10.0) + 1;
			}
		}
	}
}

void init_matrix_rectangle(PFT A, int m, int n, int flag)
{
	init_matrix_offset(A, m, n, 0, 0, flag);
}

void init_matrix(PFT A, int n, int flag)
{
	init_matrix_offset(A, n, n, 0, 0, flag);
}

int write_matrix(PFT A, int m, int n, FILE* fh, long offset)
{
	if(offset != 0)
		fseek(fh, offset, SEEK_SET);
	long long size = (long long)m * (long long)n * 8L;
	char* ptr = (char*)A;
	while(size > 0)
	{
		fwrite(ptr, sizeof(char), ((size > WRITE_BLOCK_MAX) ? WRITE_BLOCK_MAX : size), fh);
		size -= WRITE_BLOCK_MAX;
	}
	
}
int read_matrix(PFT M, int m, int n, FILE* fh, long offset)
{
	if(offset != 0)
		fseek(fh, offset, SEEK_SET);
	long long size = (long long)m * (long long)n;
	char* ptr = (char*)M;
	while(size > 0)
	{
		fread(ptr, sizeof(char), ((size > READ_BLOCK_MAX) ? READ_BLOCK_MAX : size), fh);
		size -= WRITE_BLOCK_MAX;
	}
}


void init_matrix_disk(MatrixDisk* m, int n, int WBC, int WA)
{
	mat_file_cache.fhA = fopen(CACHE_FILE_NAME_A, "wb");
	mat_file_cache.fhB = fopen(CACHE_FILE_NAME_B, "wb");
	#pragma omp parallel num_threads(3)
	{
		#pragma omp sections nowait
		{
			#pragma omp section //init matirx A
			{
				init_matrix_rectangle(m->A, n, WA, 1);
				write_matrix(m->A, n, WA, mat_file_cache.fhA, 0);
				int i, wt;
				int wb = (n+WA-1)/WA, rw = n % WA;
				for(i = 0; i < wb-1; ++i)
				{
					wt = (i != wb-2 || !rw) ? WA : rw;
					init_matrix_offset(m->A2, n, wt, 0, (i+1)*WA, 1);
					write_matrix(m->A2, n, wt, mat_file_cache.fhA, 0);
				}
			}
			#pragma omp section //init matrix B
			{
				init_matrix_rectangle(m->B, n, WBC, 1);
				write_matrix(m->B, n, WBC, mat_file_cache.fhB, 0);
				int i, wt;
				int wb = (n+WBC-1)/WBC, rw = n % WBC;
				for(i = 0; i < wb-1; ++i)
				{
					wt = (i != wb-2 || !rw) ? WBC : rw;
					init_matrix_offset(m->B2, n, wt, 0, (i+1)*WBC, 1);
					write_matrix(m->B2, n, wt, mat_file_cache.fhB, 0);
				}
			}
			#pragma omp section
			{
				init_matrix_rectangle(m->C, n, WBC, 0);
				init_matrix_rectangle(m->C2, n, WBC, 0);
			}
		}
	}
	fclose(mat_file_cache.fhA);
	mat_file_cache.fhA = NULL;
	fclose(mat_file_cache.fhB);
	mat_file_cache.fhB = NULL;
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

int checkMatrixSize(int n)
{
	return (double)n * (double)n * 3 * 8 * SWELL_FACTOR < MEMORY_SIZE;
}

void matrix_in_memory(int n)
{
	int i;
	int loopcnt = 1;
	double tc, tc2;

	
	PFT A = _mm_malloc(n*n*sizeof(FT), 32);
	PFT B = _mm_malloc(n*n*sizeof(FT), 32);
	PFT C = _mm_malloc(n*n*sizeof(FT), 32);

	init_matrix(A, n, 0);
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
	A = NULL;
	_mm_free(B);
	B = NULL;
	_mm_free(C);
	C = NULL;

}

void matrix_in_memory_disk(int n)
{
	// partition C, B, W need to be divided by NC
	int WBC = MEMORY_SIZE/(8L * 6L * n); //width of C, B to calculate 
	//int mWb = (WBC/NC >= 2) ? ((WBC/NC)*n*sizeof(FT) <= MALLOC_MAX_SIZE ? WBC/NC : -1) : 2; //if mWb < 2, the matrix should be calculated in memory
	int mWb = ((WBC/NC)*n*sizeof(FT) <= MALLOC_MAX_SIZE) ? WBC/NC : -1;
	//if mWb < 2, it should be calculated in memory
	if(mWb < 2)
	{
		printf("Should be calculated in memory, or Matrix is too large, please change n or NC\n\n");
		return;
	}
	WBC = mWb * NC; //real W

	// partition A, WA need to be divided by KC, and WA >= 2*KC
	int WA = (WBC/KC)*KC;

#ifdef PRINT_DEBUG
	printf("parameters ==> n: %d, WBC: %d, mWb: %d, WA: %d\n\n", n, WBC, mWb, WA);
#endif
	MatrixDisk md;

	/*mat_disk.A = _mm_malloc(WA*n*sizeof(FT), 32);
	mat_disk.B = _mm_malloc(WBC*n*sizeof(FT), 32);
	mat_disk.C = _mm_malloc(WBC*n*sizeof(FT), 32);

	mat_disk.A2 = _mm_malloc(WA*n*sizeof(FT), 32);
	mat_disk.B2 = _mm_malloc(WBC*n*sizeof(FT), 32);
	mat_disk.C2 = _mm_malloc(WBC*n*sizeof(FT), 32);*/
	
	md.A = _mm_malloc(WA*n*sizeof(FT), 32);
	md.B = _mm_malloc(WBC*n*sizeof(FT), 32);
	md.C = _mm_malloc(WBC*n*sizeof(FT), 32);

	md.A2 = _mm_malloc(WA*n*sizeof(FT), 32);
	md.B2 = _mm_malloc(WBC*n*sizeof(FT), 32);
	md.C2 = _mm_malloc(WBC*n*sizeof(FT), 32);
#ifdef PRINT_DEBUG
	printf("init, %p %p %p %p %p %p\n", md.A, md.B, md.C, md.A2, md.B2, md.C2);
	printf("size, %d %d %d\n", sizeof(md.A), sizeof(md.B), sizeof(md.C));
#endif

	mat_disk.A = md.A;
	mat_disk.B = md.B;
	mat_disk.C = md.C;
	mat_disk.A2 = md.A2;
	mat_disk.B2 = md.B2;
	mat_disk.C2 = md.C2;
#ifdef PRINT_DEBUG
	printf("init, %p %p %p %p %p %p\n", mat_disk.A, mat_disk.B, mat_disk.C, mat_disk.A2, mat_disk.B2, mat_disk.C2);
	printf("size, %d %d %d\n", sizeof(mat_disk.A), sizeof(mat_disk.B), sizeof(mat_disk.C));
#endif

	init_matrix_disk(&mat_disk, n,WBC,WA);
	
	//lock, so the first lock in IO loop will wait
	omp_set_lock(&read_lock);
	//exchange A/A2, B/B2, C/C2 for mat_mat_mul_disk, because every time wac == 0, those matrix will be exchanged
	PFT t = mat_disk.A;
	mat_disk.A = mat_disk.A2;
	mat_disk.A2 = t;
	t = mat_disk.B;
	mat_disk.B = mat_disk.B2;
	mat_disk.B2 = t;
	t = mat_disk.C;
	mat_disk.C = mat_disk.C2;
	mat_disk.C2 = t;


	//open nested
	omp_set_nested(1);
	#pragma omp parallel num_threads(2)
	{
		#pragma omp sections 
		{
			#pragma omp section // I/O thread
			{
				//open file
				mat_file_cache.fhA = fopen(CACHE_FILE_NAME_A, "rb");
				mat_file_cache.fhB = fopen(CACHE_FILE_NAME_B, "rb");
				mat_file_cache.fhC = fopen(CACHE_FILE_NAME_C, "wb");
				//fseek B, go over the first panel WBC*n
				fseek(mat_file_cache.fhB, WBC * n * sizeof(FT), SEEK_SET);
				//fseek A, go over the first panel WA*n
				fseek(mat_file_cache.fhA, WA * n * sizeof(FT), SEEK_SET);
				int j, wc, wa, c_in_b2, c_in_c2, cc, k, i_b, i_c;
				int wb = (n+WBC-1)/WBC, rw = n % WBC;
				int wab = (n + WA -1 )/WA, rwa = n % WA; //WA must be divided by KC
				//int we = (rwa != 0) ? (WBC + wab - 2) / (wab - 1) : (WBC + wab -1)/wab;
				int rbbn = wab*2-2; //times of reading panel of panel B
				int we = (WBC + rbbn - 1)/(rbbn);
				int nb, i_nb;
				int run_flag = 1;
				
				
				
				i_b = 1; // the first B panel to read is 1th
				i_c = 0; // the first C panel to write is 0th

				wc = WBC;
				nb = WBC/NC;// WBC can be divided by NC
				i_nb = 0;

				c_in_b2 = 0;// columns in b2 
				c_in_c2 = 0;// columns in c2
				k = 0;
				PFT pB = mat_disk.B2;
				PFT pC = mat_disk.C2;
				while(run_flag)
				{
					omp_set_lock(&read_lock);
					//lock change
					omp_set_lock(&change_lock);
#ifdef PRINT_DEBUG
					printf("\n Begin IO\n");
#endif
				
					wa = (READ_A_PANEL_INDEX != wab - 1 || !rwa) ? WA : rwa;
#ifdef PRINT_DEBUG
					printf("read panel of A, index:%d, row:%d, cl:%d\n", READ_A_PANEL_INDEX, n, wa);
#endif
					if(READ_A_PANEL_INDEX == 0) // reset the file pointer to the beginning
						fseek(mat_file_cache.fhA, 0, SEEK_SET);
					//read panel of A to A2
					read_matrix(&mat_disk.A2[READ_A_PANEL_INDEX*WA*n], n, wa, mat_file_cache.fhA, 0);

					//read panel of panel of B to B2 and write panel of panel of C to File
					if(c_in_b2 < wc && READ_A_PANEL_INDEX != wab-1) // do not read/write for the last panel of A
					{
						k = (c_in_b2 + we <= wc) ? we : (wc - c_in_b2);
						if(i_b < wb) // the last iteration do not need read B
						{
#ifdef PRINT_DEBUG
							printf("read panel of panel of B, from %d, row:%d, cl:%d, to fileB\n", c_in_b2, n, k);
#endif
							read_matrix(&mat_disk.B2[c_in_b2*n], n, k, mat_file_cache.fhB, 0);

						}
						c_in_b2 += k;
						
					}
					if(c_in_c2 < WBC && READ_A_PANEL_INDEX != wab-1 && i_b > 1)
					{
						k = (c_in_c2 + we < WBC) ? we : (WBC - c_in_c2);
#ifdef PRINT_DEBUG
						printf("write and memset panel of panel of C, from %d, row:%d, cl:%d, to fileC\n", c_in_c2, n, k);
#endif
						write_matrix(&mat_disk.C2[c_in_c2*n], n, k, mat_file_cache.fhC, 0);
						memset((void*)(&mat_disk.C2[c_in_c2*n]), 0, n*k*sizeof(FT));
						c_in_c2 += k;
					}

					

					if(READ_A_PANEL_INDEX == 0)
					{
						++ i_nb;// finish reading one A 
						if(i_nb == nb)
						{
							if(i_b > 1)
								++i_c;
							++ i_b;
							wc = (i_b != wb -1 || !rw) ? WBC : rw;
							c_in_b2 = 0;
							c_in_c2 = 0;
							i_nb = 0;
#ifdef PRINT_DEBUG
							printf("next iteration, i_c:%d, i_b/wb: %d/%d, wc: %d\n", i_c, i_b, wb, wc);
#endif
							if(i_c == wb - 1)
							{
								run_flag = 0;
							}
						}
					}
					omp_unset_lock(&change_lock);
				}
				
				printf("finish calculating!!!!!!!!!\n");
				//fflush(mat_file_cache.fhA);
				//fflush(mat_file_cache.fhB);
				//fflush(mat_file_cache.fhC);
				//fclose(mat_file_cache.fhA);
				//fclose(mat_file_cache.fhB);
				//fclose(mat_file_cache.fhC);

			}

			#pragma omp section // mat_mul
			{
				int i, wc;
				PFT tmp;
				int wb = (n + WBC - 1)/WBC, rw = n % WBC;
				double tc = dsecnd();

				for(i = 0;i < wb; ++i)
				{
					wc = (i != wb-1 || !rw) ? WBC : rw;
#ifdef PRINT_DEBUG
					printf("start multiply panel matrix: n:%d, WBC:%d, i/wb: %d/%d\n", n, WBC, i, wb);
#endif
					// matrix exchange happened in mat_mat_mul_disk funciton
					mat_mat_mul_disk(&mat_disk, n, wc, WA);
#ifdef PRINT_DEBUG
					printf("finish multiply panel matrix: n:%d, WBC:%d, i/wb: %d/%d\n", n, WBC, i, wb);
#endif
						
#ifdef PRINT_DEBUG
	printf("status md, %p %p %p %p %p %p\n", md.A, md.B, md.C, md.A2, md.B2, md.C2);
	printf("status mt, %p %p %p %p %p %p\n", mat_disk.A, mat_disk.B, mat_disk.C, mat_disk.A2, mat_disk.B2, mat_disk.C2);
#endif
				}
				double tc2 = dsecnd();

				printf("disk mmt time : %.5f milliseconds, %.5f GFlops \n\n", ((tc2-tc))*1000.0, (2.0*n*n*n)/(tc2-tc));

			}

		}
	}
#ifdef PRINT_DEBUG
	printf("begin write last panel of C\n");
#endif

	//write last panel C;
	int lpw = (n % WBC == 0) ? WBC : (n % WBC);
	write_matrix(mat_disk.C, n, lpw, mat_file_cache.fhC, 0);
	fflush(mat_file_cache.fhA);
	fflush(mat_file_cache.fhB);
	fflush(mat_file_cache.fhC);
	fclose(mat_file_cache.fhA);
	fclose(mat_file_cache.fhB);
	fclose(mat_file_cache.fhC);

#ifdef PRINT_DEBUG
	printf("A: %.1f %.1f %.1f\n", mat_disk.A[0], mat_disk.A[1], mat_disk.A[2]);
	printf("B: %.1f %.1f %.1f\n", mat_disk.B[0], mat_disk.B[1], mat_disk.B[2]);
	printf("C: %.1f %.1f %.1f\n", mat_disk.C[0], mat_disk.C[1], mat_disk.C[2]);
	printf("A2: %.1f %.1f %.1f\n", mat_disk.A2[0], mat_disk.A2[1], mat_disk.A2[2]);
	printf("B2: %.1f %.1f %.1f\n", mat_disk.B2[0], mat_disk.B2[1], mat_disk.B2[2]);
	printf("C2: %.1f %.1f %.1f\n", mat_disk.C2[0], mat_disk.C2[1], mat_disk.C2[2]);
#endif

#ifdef PRINT_DEBUG
	printf("free A, %p %p %p %p %p %p\n", md.A, md.B, md.C, md.A2, md.B2, md.C2);
#endif
	_mm_free(md.A);
	md.A = NULL;
#ifdef PRINT_DEBUG
	printf("free B, %p %p %p %p %p %p\n", mat_disk.A, mat_disk.B, mat_disk.C, mat_disk.A2, mat_disk.B2, mat_disk.C2);
#endif
	_mm_free(md.B);
	md.B = NULL;
#ifdef PRINT_DEBUG
	printf("free C, %p %p %p %p %p %p\n", mat_disk.A, mat_disk.B, mat_disk.C, mat_disk.A2, mat_disk.B2, mat_disk.C2);
#endif
	_mm_free(md.C);
	md.C = NULL;
#ifdef PRINT_DEBUG
	printf("free A2, %p %p %p %p %p %p\n", mat_disk.A, mat_disk.B, mat_disk.C, mat_disk.A2, mat_disk.B2, mat_disk.C2);
#endif
	
	_mm_free(md.A2);
	md.A2 = NULL;
#ifdef PRINT_DEBUG
	printf("free B2, %p %p %p %p %p %p\n", mat_disk.A, mat_disk.B, mat_disk.C, mat_disk.A2, mat_disk.B2, mat_disk.C2);
#endif
	_mm_free(md.B2);
	md.B2 = NULL;
#ifdef PRINT_DEBUG
	printf("free C2, %p %p %p %p %p %p\n", mat_disk.A, mat_disk.B, mat_disk.C, mat_disk.A2, mat_disk.B2, mat_disk.C2);
#endif
	_mm_free(md.C2);
	md.C2 = NULL;

}

int main()
{
	int n = 8192;
	
	if(checkMatrixSize(n)) //matrix can be fit into memory;
	{
		matrix_in_memory(n);
	}
	else // need to use disk as temporary cache
	{
		omp_init_lock(&read_lock);
		omp_init_lock(&change_lock);
		matrix_in_memory_disk(n);
		omp_destroy_lock(&read_lock);
		omp_destroy_lock(&change_lock);
	}

	

	return 0;
}

