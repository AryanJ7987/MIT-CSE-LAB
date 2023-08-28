#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define BUFSIZE 256

void Error_Handler(int error_code) {
    if (error_code != MPI_SUCCESS) {
        char error_string[BUFSIZE];
        int length_of_error_string, error_class;
        MPI_Error_class(error_code, &error_class);
        MPI_Error_string(error_code, error_string, &length_of_error_string);
        fprintf(stderr, "[ERROR %d] : %s\n", error_class, error_string);
        MPI_Abort(MPI_COMM_WORLD, error_code);
    }
}

int main(int argc, char *argv[]) {
    int rank, size, root = 0, mpi_err;
    int matrix[3][3];
    int search_element;
    int local_count = 0, global_count = 0;

    mpi_err = MPI_Init(&argc, &argv);
    Error_Handler(mpi_err);

    mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Error_Handler(mpi_err);
    mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &size);
    Error_Handler(mpi_err);

    if (rank == root) {
        printf("Enter the 3x3 matrix:\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }

        printf("Enter an element to search: ");
        scanf("%d", &search_element);
    }

    // Broadcast the search element to all processes
    mpi_err = MPI_Bcast(&search_element, 1, MPI_INT, root, MPI_COMM_WORLD);
    Error_Handler(mpi_err);

    // Each process searches its portion of the matrix
    for (int i = rank; i < 3; i += size) {
        for (int j = 0; j < 3; j++) {
            if (matrix[i][j] == search_element) {
                local_count++;
            }
        }
    }

    // Sum up local counts to get the global count
    mpi_err = MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
    Error_Handler(mpi_err);

    if (rank == root) {
        printf("Number of occurrences of %d in the matrix: %d\n", search_element, global_count);
    }

    mpi_err = MPI_Finalize();
    Error_Handler(mpi_err);

    return 0;
}
