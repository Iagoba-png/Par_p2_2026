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
  int *local_data1, *local_data2;
  int *local_result;
  
  int rank, size;
  int rows;  // Number of rows per process
  int remainder;
  int local_rows;
  
  double comm_start, comm_end, comp_start, comp_end;
  double comm_time = 0.0, comp_time = 0.0;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // Calculate rows per process
  rows = M / size;
  remainder = M % size;
  
  // Determine local number of rows for this process
  // Processes with rank < remainder get one extra row
  if (rank < remainder) {
    local_rows = rows + 1;
  } else {
    local_rows = rows;
  }
  
  // Allocate memory for local data
  local_data1 = (int *) malloc(local_rows * N * sizeof(int));
  local_data2 = (int *) malloc(local_rows * N * sizeof(int));
  local_result = (int *) malloc(local_rows * sizeof(int));
  
  // Process 0 initializes the full matrices
  if (rank == 0) {
    data1 = (int *) malloc(M * N * sizeof(int));
    data2 = (int *) malloc(M * N * sizeof(int));
    result = (int *) malloc(M * sizeof(int));
    
    /* Initialize Matrices */
    for(i=0; i<M; i++) {
      for(j=0; j<N; j++) {
        data1[i*N+j] = fast_rand();
        data2[i*N+j] = fast_rand();
      }
    }
  }
  
  // Prepare send counts and displacements for Scatterv
  int *sendcounts = NULL;
  int *displs = NULL;
  
  if (rank == 0) {
    sendcounts = (int *) malloc(size * sizeof(int));
    displs = (int *) malloc(size * sizeof(int));
    
    int offset = 0;
    for(i=0; i<size; i++) {
      int proc_rows = (i < remainder) ? (rows + 1) : rows;
      sendcounts[i] = proc_rows * N;
      displs[i] = offset;
      offset += proc_rows * N;
    }
  }
  
  // Start communication time
  comm_start = MPI_Wtime();
  
  // Distribute data1 and data2 to all processes
  MPI_Scatterv(rank == 0 ? data1 : NULL, sendcounts, displs, MPI_INT,
               local_data1, local_rows * N, MPI_INT,
               0, MPI_COMM_WORLD);
               
  MPI_Scatterv(rank == 0 ? data2 : NULL, sendcounts, displs, MPI_INT,
               local_data2, local_rows * N, MPI_INT,
               0, MPI_COMM_WORLD);
  
  comm_end = MPI_Wtime();
  comm_time += (comm_end - comm_start);
  
  // Start computation time
  comp_start = MPI_Wtime();
  
  // Each process computes its local portion of the result
  for(i=0; i<local_rows; i++) {
    local_result[i] = 0;
    for(j=0; j<N; j++) {
      local_result[i] += base_distance(local_data1[i*N+j], local_data2[i*N+j]);
    }
  }
  
  comp_end = MPI_Wtime();
  comp_time += (comp_end - comp_start);
  
  // Prepare receive counts and displacements for Gatherv
  int *recvcounts = NULL;
  int *recvdispls = NULL;
  
  if (rank == 0) {
    recvcounts = (int *) malloc(size * sizeof(int));
    recvdispls = (int *) malloc(size * sizeof(int));
    
    int offset = 0;
    for(i=0; i<size; i++) {
      int proc_rows = (i < remainder) ? (rows + 1) : rows;
      recvcounts[i] = proc_rows;
      recvdispls[i] = offset;
      offset += proc_rows;
    }
  }
  
  // Start communication time for gathering results
  comm_start = MPI_Wtime();
  
  // Gather results to process 0
  MPI_Gatherv(local_result, local_rows, MPI_INT,
              rank == 0 ? result : NULL, recvcounts, recvdispls, MPI_INT,
              0, MPI_COMM_WORLD);
  
  comm_end = MPI_Wtime();
  comm_time += (comm_end - comm_start);
  
  // Print times for each process
  for(i=0; i<size; i++) {
    if(rank == i) {
      printf("Process %d - Communication time: %lf seconds\n", rank, comm_time);
      printf("Process %d - Computation time: %lf seconds\n", rank, comp_time);
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  /* Display result (only process 0) */
  if (rank == 0) {
    if (DEBUG == 1) {
      int checksum = 0;
      for(i=0; i<M; i++) {
        checksum += result[i];
      }
      printf("Checksum: %d\n", checksum);
    } else if (DEBUG == 2) {
      for(i=0; i<M; i++) {
        printf(" %d \t ", result[i]);
      }
      printf("\n");
    }
    
    free(data1); 
    free(data2); 
    free(result);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(recvdispls);
  }
  
  free(local_data1); 
  free(local_data2); 
  free(local_result);
  
  MPI_Finalize();

  return 0;
}
