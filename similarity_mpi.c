#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

#define DEBUG 0

/* Translation of the DNA bases
   A -> 0
   C -> 1
   G -> 2
   T -> 3
   N -> 4*/

#define M  1000000 // Number of sequences
#define N  200     // Number of bases per sequence

unsigned int g_seed = 0;

int fast_rand(void) {
    g_seed = (214013*g_seed+2531011);
    return (g_seed>>16) % 5;
}

// The distance between two bases
int base_distance(int base1, int base2){

  if((base1 == 4) || (base2 == 4)){
    return 3;
  }

  if(base1 == base2) {
    return 0;
  }

  if((base1 == 0) && (base2 == 3)) {
    return 1;
  }

  if((base2 == 0) && (base1 == 3)) {
    return 1;
  }

  if((base1 == 1) && (base2 == 2)) {
    return 1;
  }

  if((base2 == 2) && (base1 == 1)) {
    return 1;
  }

  return 2;
}

int main(int argc, char *argv[] ) {

  int i, j;
  int *data1, *data2;
  int *result;
  int *loc_data1, *loc_data2;
  int *loc_result;

  int rank, size;
  int filas_gen;
  int resto;
  int filas_conc;

  double comm_start, comm_end, comp_start, comp_end;
  double comm_time = 0.0, comp_time = 0.0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  filas_gen = M / size;
  resto = M % size;
  
  if (rank < resto) {
    filas_conc = filas_gen + 1;
  } else {
    filas_conc = filas_gen;
  }

  loc_data1 = (int *) malloc(filas_conc * N * sizeof(int));
  loc_data2 = (int *) malloc(filas_conc * N * sizeof(int));
  loc_result = (int *) malloc(filas_conc * sizeof(int));

  if (rank == 0) {
    data1 = (int *) malloc(M * N * sizeof(int));
    data2 = (int *) malloc(M * N * sizeof(int));
    result = (int *) malloc(M * sizeof(int));

    for(i=0; i<M; i++) {
      for(j=0; j<N; j++) {
        data1[i*N+j] = fast_rand();
        data2[i*N+j] = fast_rand();
      }
    }
  }

  int *sendcounts = NULL;
  int *despl = NULL;

  if (rank == 0) {
    sendcounts = (int *) malloc(size * sizeof(int));
    despl = (int *) malloc(size * sizeof(int));

    int offset = 0;
    for(i=0; i<size; i++) {
      int filas_func = (i < resto) ? (filas_gen + 1) : filas_gen;
      sendcounts[i] = filas_func * N;
      despl[i] = offset;
      offset += filas_func * N;
    }
  }

  comm_start = MPI_Wtime();

  MPI_Scatterv(rank == 0 ? data1 : NULL, sendcounts, despl, MPI_INT,
               loc_data1, filas_conc * N, MPI_INT,
               0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? data2 : NULL, sendcounts, despl, MPI_INT,
               loc_data2, filas_conc * N, MPI_INT,
               0, MPI_COMM_WORLD);

  comm_end = MPI_Wtime();
  comm_time += (comm_end - comm_start);

  comp_start = MPI_Wtime();

  for(i=0; i<filas_conc; i++) {
    loc_result[i] = 0;
    for(j=0; j<N; j++) {
      loc_result[i] += base_distance(loc_data1[i*N+j], loc_data2[i*N+j]);
    }
  }

  comp_end = MPI_Wtime();
  comp_time += (comp_end - comp_start);

  int *recvcounts = NULL;
  int *recvdespl = NULL;

  if (rank == 0) {
    recvcounts = (int *) malloc(size * sizeof(int));
    recvdespl = (int *) malloc(size * sizeof(int));

    int offset = 0;
    for(i=0; i<size; i++) {
      int filas_func = (i < resto) ? (filas_gen + 1) : filas_gen;
      recvcounts[i] = filas_func;
      recvdespl[i] = offset;
      offset += filas_func;
    }
  }

  comm_start = MPI_Wtime();

  MPI_Gatherv(loc_result, filas_conc, MPI_INT,
              rank == 0 ? result : NULL, recvcounts, recvdespl, MPI_INT,
              0, MPI_COMM_WORLD);

  comm_end = MPI_Wtime();
  comm_time += (comm_end - comm_start);

  for(i=0; i<size; i++) {
    if(rank == i) {
      printf("Process %d - Communication time: %lf seconds\n", rank, comm_time);
      printf("Process %d - Computation time: %lf seconds\n", rank, comp_time);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (rank == 0) {
      int checksum = 0;
      for(i=0; i<M; i++) {
        checksum += result[i];
      }
      printf("Checksum: %d\n", checksum);

    free(data1);
    free(data2);
    free(result);
    free(sendcounts);
    free(despl);
    free(recvcounts);
    free(recvdespl);
  }

  free(loc_data1);
  free(loc_data2);
  free(loc_result);
  
  MPI_Finalize();

  return 0;
}
