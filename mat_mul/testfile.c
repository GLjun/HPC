/*************************************************************************
    > File Name: testfile.c
    > Author: cgn
    > Func: 
    > Created Time: 二 12/19 13:03:44 2017
 ************************************************************************/

#include <stdio.h>

void test()
{
	FILE* f = fopen("1.txt", "w");
	fseek(f, 10, SEEK_SET);
	fputs("abcdefghij", f);
	fseek(f, 0, SEEK_SET);
	fputs("abcdefghij", f);
	fclose(f);
}
int main()
{
	test();
	return 0;
}
