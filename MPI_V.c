#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Example data to send
    int send_data = rank;
    int *recv_data_allgather = (int *)malloc(size * sizeof(int));
    int *recv_data_allgatherv = (int *)malloc(size * sizeof(int));
    int *send_data_alltoall = (int *)malloc(size * sizeof(int));
    int *recv_data_alltoall = (int *)malloc(size * sizeof(int));
    int *send_counts = (int *)malloc(size * sizeof(int));
    int *recv_counts = (int *)malloc(size * sizeof(int));
    int *send_displs = (int *)malloc(size * sizeof(int));
    int *recv_displs = (int *)malloc(size * sizeof(int));
    int *send_data_alltoallv = (int *)malloc(size * sizeof(int));
    int *recv_data_alltoallv = (int *)malloc(size * sizeof(int));

    // Prepare data for allgatherv
    int send_count = 1;
    for (int i = 0; i < size; ++i) {
        send_counts[i] = 1;
        recv_counts[i] = 1;
        send_displs[i] = i;
        recv_displs[i] = i;
    }

    // MPI_Allgather
    MPI_Allgather(&send_data, 1, MPI_INT, recv_data_allgather, 1, MPI_INT, MPI_COMM_WORLD);

    // MPI_Allgatherv
    MPI_Allgatherv(&send_data, send_count, MPI_INT, recv_data_allgatherv, recv_counts, recv_displs, MPI_INT, MPI_COMM_WORLD);

    // Prepare data for alltoall
    for (int i = 0; i < size; ++i) {
        send_data_alltoall[i] = rank + i;
        send_data_alltoallv[i] = rank * size + i;
    }

    // MPI_Alltoall
    MPI_Alltoall(send_data_alltoall, 1, MPI_INT, recv_data_alltoall, 1, MPI_INT, MPI_COMM_WORLD);

    // Prepare data for alltoallv
    for (int i = 0; i < size; ++i) {
        send_counts[i] = i + 1; // Varying count for each process
        recv_counts[i] = rank + 1; // Varying count for each process
        send_displs[i] = i;
        recv_displs[i] = rank;
    }

    // MPI_Alltoallv
    MPI_Alltoallv(send_data_alltoallv, send_counts, send_displs, MPI_INT, recv_data_alltoallv, recv_counts, recv_displs, MPI_INT, MPI_COMM_WORLD);

    // Print the results
    printf("Rank %d: MPI_Allgather result: ", rank);
    for (int i = 0; i < size; ++i) {
        printf("%d ", recv_data_allgather[i]);
    }
    printf("\n");

    printf("Rank %d: MPI_Allgatherv result: ", rank);
    for (int i = 0; i < size; ++i) {
        printf("%d ", recv_data_allgatherv[i]);
    }
    printf("\n");

    printf("Rank %d: MPI_Alltoall result: ", rank);
    for (int i = 0; i < size; ++i) {
        printf("%d ", recv_data_alltoall[i]);
    }
    printf("\n");

    printf("Rank %d: MPI_Alltoallv result: ", rank);
    for (int i = 0; i < size; ++i) {
        printf("%d ", recv_data_alltoallv[i]);
    }
    printf("\n");

    // Free allocated memory
    free(recv_data_allgather);
    free(recv_data_allgatherv);
    free(send_data_alltoall);
    free(recv_data_alltoall);
    free(send_counts);
    free(recv_counts);
    free(send_displs);
    free(recv_displs);
    free(send_data_alltoallv);
    free(recv_data_alltoallv);

    MPI_Finalize();
    return 0;
}
