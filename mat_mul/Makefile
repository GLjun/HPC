default : matmul
CC = icc
CFLAGS = -O3 -qopenmp -mkl -std=c99 -ipo
CLIBS = -qopenmp -mkl -std=c99

matmul : mat_mat_mul.o
	$(CC) $(CLIBS) mat_mat_mul.o -o matmul

mat_mat_mul.o : mat_mat_mul.c
	$(CC) $(CFLAGS) -c mat_mat_mul.c -o mat_mat_mul.o

clean : 
	rm -rf ./*.o matmul

run :
	srun -N1 -l ./matmul

