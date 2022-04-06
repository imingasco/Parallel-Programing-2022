matrix.out: matrix.h matrix.c main.c
	gcc --std=c99 -O2 -fopenmp matrix.h matrix.c main.c -o matrix -lm
clean:
	rm -f matrix.out
