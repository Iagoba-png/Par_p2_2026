# Makefile para compilar similarity (secuencial y paralelo)

CC = gcc
MPICC = mpicc
CFLAGS = -O3 -Wall

all: similarity similarity_mpi

similarity: similarity.c
	$(CC) $(CFLAGS) -o similarity similarity.c

similarity_mpi: similarity_mpi.c
	$(MPICC) $(CFLAGS) -o similarity_mpi similarity_mpi.c

clean:
	rm -f similarity similarity_mpi

.PHONY: all clean
