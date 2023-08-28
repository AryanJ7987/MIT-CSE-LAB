#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

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

int main(int argc, char *argv[])
{
    int rank, size, mpi_err;
    int mat[4][4], out[4][4];

    mpi_err = MPI_Init(&argc, &argv);
    Error_Handler(mpi_err);

    mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &size);
    Error_Handler(mpi_err);
    mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Error_Handler(mpi_err);

    if(rank == 0) {
        printf("Enter 4x4 Matrix : \n");
        
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < 4; j++) {
                scanf("%d", &mat[i][j]);
                out[i][j] = 0;
            }
        }
    } 

    for(int i = 0; i < 4; i++) {
        mpi_err = MPI_Bcast(mat[i], 4, MPI_INT, 0, MPI_COMM_WORLD);
        Error_Handler(mpi_err);
        mpi_err = MPI_Bcast(out[i], 4, MPI_INT, 0, MPI_COMM_WORLD);
        Error_Handler(mpi_err);
    }

    mpi_err = MPI_Scan(mat[rank], out[rank], 4, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    Error_Handler(mpi_err);
    mpi_err = MPI_Gather(out[rank], 4, MPI_INT, mat[rank], 4, MPI_INT, 0, MPI_COMM_WORLD);
    Error_Handler(mpi_err);

    if(rank == 0) {
        printf("The New Matrix is : \n");
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < 4; j++) {
                printf("%d ", mat[i][j]);
            }
            printf("\n");
        }
    }

    mpi_err = MPI_Finalize();
    Error_Handler(mpi_err);
}
