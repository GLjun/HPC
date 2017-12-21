/*************************************************************************
    > File Name: test.c
    > Author: cgn
    > Func: 
    > Created Time: 2017年12月17日 星期日 02时36分07秒
 ************************************************************************/

#include <stdio.h>
#include "mkl.h"

const long long BLOCK_SIZE = 4096LL;
const long long FILE_SIZE = 1073741824LL;

void testwrite(const char* filename)
{
	char* buffer = (char*)malloc(BLOCK_SIZE);
	FILE* f = fopen(filename, "wb");
	int i;
	double tc1 = dsecnd();
	for(i = 0;i < (FILE_SIZE/BLOCK_SIZE); ++i)
	{

		fwrite(buffer, sizeof(char), BLOCK_SIZE, f);
	}
	double tc2 = dsecnd();
	printf("write size : %lld, time : %.5f, speed : %.5f\n\n", FILE_SIZE, (tc2-tc1)*1000, FILE_SIZE*1.0/(tc2-tc1));
	fflush(f);
	fclose(f);

	free(buffer);
}

void testread(const char* filename)
{
	char* buffer = (char*)malloc(BLOCK_SIZE);
	FILE* f = fopen(filename, "rb");
	int i;

	double tc1 = dsecnd();
	for(i = 0;i < (FILE_SIZE/BLOCK_SIZE); ++i)
	{

		fread(buffer, sizeof(char), BLOCK_SIZE, f);
	}
	double tc2 = dsecnd();
	printf("read size : %lld, time : %.5f, speed : %.5f\n\n", FILE_SIZE, (tc2-tc1)*1000, FILE_SIZE*1.0/(tc2-tc1));
	fflush(f);
	fclose(f);

	free(buffer);
}

int main()
{
	const char* filename = "1.txt";
	testwrite(filename);
	testread(filename);
	return 0;

}



