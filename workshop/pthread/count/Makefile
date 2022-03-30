all: slow fast private

slow:
	$(CC) -std=c17 main.c -c
	$(CC) -std=c17 sum.c -c
	$(CC) -std=c17 main.o sum.o job.c -o slow -lpthread

fast:
	$(CC) -std=c17 main.c -c
	$(CC) -std=c17 sum.c -c
	$(CC) -std=c17 main.o sum.o job_fast.c -o fast -lpthread

private:
	$(CC) -std=c17 main.c -c
	$(CC) -std=c17 sum_private.c -c
	$(CC) -std=c17 main.o sum_private.o job.c -o private -lpthread

clean:
	rm -f *.o slow fast private
