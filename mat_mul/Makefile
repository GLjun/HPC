default : matmul
CC = icc
CFLAGS = -O3 -qopenmp -mkl -std=c99 -ipo -g
CLIBS = -qopenmp -mkl -std=c99  -g

matmul : disk_mat_mat_mul.o
	$(CC) $(CLIBS) disk_mat_mat_mul.o -o matmul

disk_mat_mat_mul.o : mat_mat_mul.c
	$(CC) $(CFLAGS)  -c diskV2_mat_mul.c -o disk_mat_mat_mul.o

clean : 
	rm -rf ./*.o matmul cacheA cacheB cacheC

run :
	srun -N1 -l ./matmul

