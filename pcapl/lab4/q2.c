#include <stdio.h>
#include <math.h>
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

double f(double x) {
    return 4.0 / (1.0 + x * x);
}

int main(int argc, char *argv[]) {
    int rank, size, mpi_err;
    double a = 0.0; // Lower bound of integration
    double b = 1.0; // Upper bound of integration
    int n = 1000000; // Number of rectangles

    mpi_err = MPI_Init(&argc, &argv);
    Error_Handler(mpi_err);

    mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Error_Handler(mpi_err);
    mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &size);
    Error_Handler(mpi_err);

    // Each process calculates its range of rectangles
    int local_n = n / size;
    double local_a = a + rank * (b - a) / size;
    double local_b = local_a + (b - a) / size;

    double local_sum = 0.0;
    for (int i = 0; i < local_n; i++) {
        double x = local_a + (i + 0.5) * (local_b - local_a) / local_n;
        local_sum += f(x);
    }

    double global_sum = 0.0;
    mpi_err = MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    Error_Handler(mpi_err);

    if (rank == 0) {
        double pi = global_sum * (b - a) / n;
        printf("Approximate value of π: %.16f\n", pi);
    }

    mpi_err = MPI_Finalize();
    Error_Handler(mpi_err);

    return 0;
}
