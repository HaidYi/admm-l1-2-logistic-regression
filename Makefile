GSLROOT=/usr/local
# use this if on 64-bit machine with 64-bit GSL libraries
ARCH=x86_64
# use this if on 32-bit machine with 32-bit GSL libraries
# ARCH=i386

MPICC=mpicc
CC=gcc
CFLAGS=-Wall -std=c99 -arch $(ARCH) -I$(GSLROOT)/include
LDFLAGS=-L$(GSLROOT)/lib -lgsl -lgslcblas -lm

all: mpi_logit logit

mpi_logit: mpi_logit.o solver.o mmio.o
	$(MPICC) $(CFLAGS) $(LDFLAGS) mmio.o solver.o mpi_logit.o -o mpi_logit

logit: logit.o solver.o mmio.o
	$(CC) $(CFLAGS) $(LDFLAGS) mmio.o solver.o logit.o -o logit

mpi_logit.o: mpi_logit.c mmio.o solver.o
	$(CC) $(CFLAGS) -c mpi_logit.c

logit.o: logit.c mmio.o solver.o
	$(CC) $(CFLAGS) -c logit.c

solver.o: solver.c 
	$(CC) $(CFLAGS) -c solver.c

mmio.o: mmio.c
	$(CC) $(CFLAGS) -c mmio.c

clean:
	rm -vf *.o logit mpi_logit 
