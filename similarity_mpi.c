/*
 * similarity_mpi.c
 * Paralelización con MPI del cálculo de similaridad entre secuencias de ADN.
 * Práctica 3 - Concurrencia e Paralelismo (Bloque II)
 *
 * Compilar con:
 *   mpicc -O2 -o similarity_mpi similarity_mpi.c
 * Ejecutar con (por ejemplo, 4 procesos):
 *   mpirun -np 4 ./similarity_mpi
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

#define DEBUG 0

/* Traducción de bases de ADN
   A -> 0
   C -> 1
   G -> 2
   T -> 3
   N -> 4 */

#define M  1000000  // Número de secuencias
#define N  200      // Número de bases por secuencia

unsigned int g_seed = 0;

int fast_rand(void) {
    g_seed = (214013 * g_seed + 2531011);
    return (g_seed >> 16) % 5;
}

/* Distancia entre dos bases */
int base_distance(int base1, int base2) {
    if ((base1 == 4) || (base2 == 4))
        return 3;
    if (base1 == base2)
        return 0;
    if ((base1 == 0 && base2 == 3) || (base1 == 3 && base2 == 0))
        return 1;
    if ((base1 == 1 && base2 == 2) || (base1 == 2 && base2 == 1))
        return 1;
    return 2;
}

int main(int argc, char *argv[]) {
    int rank, size, proporcionado;
    int i, j;
    int *data1 = NULL, *data2 = NULL;   // Solo completas en proceso 0
    int *local_data1, *local_data2;     // Parte local de cada proceso
    int *result = NULL;                 // Vector completo en proceso 0
    int *local_result;                  // Resultado parcial de cada proceso
    double t_start, t_end;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &proporcionado);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ----- 1. Proceso 0 genera los datos secuencialmente ----- */
    if (rank == 0) {
        data1 = (int *) malloc(M * N * sizeof(int));
        data2 = (int *) malloc(M * N * sizeof(int));
        result = (int *) malloc(M * sizeof(int));

        g_seed = 0;  // reiniciar semilla (ya es 0, pero por claridad)
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                data1[i * N + j] = fast_rand();
                data2[i * N + j] = fast_rand();
            }
        }
    }

    /* ----- 2. Reparto de filas entre procesos ----- */
    // Calcular cuántas filas le corresponden a cada proceso
    int *sendcounts = NULL, *displs = NULL;
    int local_M;   // filas que procesará este proceso

    if (rank == 0) {
        sendcounts = (int *) malloc(size * sizeof(int));
        displs = (int *) malloc(size * sizeof(int));
    }

    // Reparto uniforme: base = M / size, resto = M % size
    int base = M / size;
    int resto = M % size;

    // El proceso 0 rellena los vectores de envío
    if (rank == 0) {
        int offset = 0;
        for (int p = 0; p < size; p++) {
            sendcounts[p] = (p < resto) ? (base + 1) : base;
            displs[p] = offset;
            offset += sendcounts[p] * N;  // en elementos (no en filas)
        }
    }

    // Cada proceso recibe su número de filas (útil para los repartos posteriores)
    // MPI_Scatter de sendcounts no está permitido directamente porque es int,
    // usamos MPI_Scatter con sendcounts como array en root 0.
    int my_rows;
    MPI_Scatter(sendcounts, 1, MPI_INT, &my_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_M = my_rows;

    // Reservar espacio para los datos locales
    local_data1 = (int *) malloc(my_rows * N * sizeof(int));
    local_data2 = (int *) malloc(my_rows * N * sizeof(int));
    local_result = (int *) malloc(my_rows * sizeof(int));

    // Distribuir data1 y data2 con Scatterv
    MPI_Scatterv(data1, sendcounts, displs, MPI_INT,
                 local_data1, my_rows * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(data2, sendcounts, displs, MPI_INT,
                 local_data2, my_rows * N, MPI_INT, 0, MPI_COMM_WORLD);

    /* ----- 3. Cálculo paralelo ----- */
    // Barrera para sincronizar antes de medir tiempo
    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    for (i = 0; i < my_rows; i++) {
        local_result[i] = 0;
        for (j = 0; j < N; j++) {
            local_result[i] += base_distance(local_data1[i * N + j],
                                             local_data2[i * N + j]);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_end = MPI_Wtime();

    /* ----- 4. Recogida de resultados en proceso 0 ----- */
    // Para Gatherv necesitamos desplazamientos en número de elementos (filas)
    int *recvcounts = NULL, *rdispls = NULL;
    if (rank == 0) {
        recvcounts = (int *) malloc(size * sizeof(int));
        rdispls = (int *) malloc(size * sizeof(int));
        int offset = 0;
        for (int p = 0; p < size; p++) {
            int filas_p = (p < resto) ? base + 1 : base;
            recvcounts[p] = filas_p;
            rdispls[p] = offset;
            offset += filas_p;
        }
    }

    MPI_Gatherv(local_result, my_rows, MPI_INT,
                result, recvcounts, rdispls, MPI_INT,
                0, MPI_COMM_WORLD);

    /* ----- 5. Mostrar resultados (solo proceso 0) ----- */
    if (rank == 0) {
        double time_sec = t_end - t_start;

        if (DEBUG == 1) {
            int checksum = 0;
            for (i = 0; i < M; i++) {
                checksum += result[i];
            }
            printf("Checksum: %d\n", checksum);
        } else if (DEBUG == 2) {
            for (i = 0; i < M; i++) {
                printf("%d\t", result[i]);
            }
            printf("\n");
        } else {
            printf("Time (seconds) = %lf\n", time_sec);
        }
    }

    /* ----- 6. Liberar memoria ----- */
    free(local_data1);
    free(local_data2);
    free(local_result);
    if (rank == 0) {
        free(data1);
        free(data2);
        free(result);
        free(sendcounts);
        free(displs);
        free(recvcounts);
        free(rdispls);
    }

    MPI_Finalize();
    return 0;
}