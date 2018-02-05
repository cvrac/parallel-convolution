#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <assert.h>
#include "mpi.h"

#define MASTER_PROCESS 0
#define DIMENSIONALITY 2


/*
 * 1. Find process row and column offsets and vector definition for the input data
 * 2. Convolution filter definition
 * 3. Parallel read of the input file "image"
 *
*/



unsigned int best_fit(int rows, int cols, int processes) {
	unsigned int rows_it, cols_it, best_fit_val = 0, min = rows + cols + 1;
	unsigned int current;

	for (rows_it = 1; rows_it != processes + 1; rows_it++) {
		if (processes % rows_it || rows % rows_it)
			continue;
		cols_it = processes / rows_it;
		if (cols % cols_it) continue;
		current = rows / rows_it + cols / cols_it;
		if (current < min) {
			min = current;
			best_fit_val = rows_it;
		}
	}
	return best_fit_val;
}

int main(int argc, char **argv) {
	int comm_sz = 0, comm_rk = 0;
	int width = 0, height = 0;
	int loops = 0, grey = 0;
	char *picture = NULL;
    int rows = 0, columns = 0;
    int proc_row = -1, proc_col = -1;
    int i = 0;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rk);

	MPI_Request se_nr_request;
	MPI_Request se_ea_request;
	MPI_Request se_so_request;
	MPI_Request se_we_request;
	MPI_Request rcv_nr_request;
	MPI_Request rcv_ea_request;
	MPI_Request rcv_so_request;	
	MPI_Request rcv_we_request;	
    MPI_Status status;
	

	MPI_Datatype send_col, send_row;

	// Cartesian topology definition
	MPI_Comm cartesianComm;
	int dims[DIMENSIONALITY] = {0, 0};
	int periods[2] = {0, 0};
	int reorder = 1;
	MPI_Dims_create(comm_sz, DIMENSIONALITY, dims);
	MPI_Cart_create(MPI_COMM_WORLD, DIMENSIONALITY, dims, periods, reorder, &cartesianComm);

	// Variables setup
	int best_fit_rows;
	if (comm_rk == MASTER_PROCESS) {
		width = atoi(argv[1]); height = atoi(argv[2]);
		loops = atoi(argv[3]);
		printf("MASTER SAYS %d\n", height);
		if (!strcmp(argv[4], "grey"))
			grey = 1;

		printf("I am the master process\n");
		best_fit_rows = best_fit(height, width, comm_sz);
		if (!best_fit_rows) {
			MPI_Abort(cartesianComm, 1);
			return 1;
		}
		printf("%d\n", best_fit_rows);
		rows = height / best_fit_rows;
		columns = width / (comm_sz / best_fit_rows);
		printf("%d %d\n", rows, columns);
	}


	picture = calloc((strlen(argv[5]) +1), sizeof(char));
	assert( picture != NULL);
	strncpy(picture, argv[5], strlen(argv[5]) + 1);

	MPI_Bcast(&width, 1, MPI_INT, MASTER_PROCESS, cartesianComm);
	MPI_Bcast(&height, 1, MPI_INT, MASTER_PROCESS, cartesianComm);
	MPI_Bcast(&loops, 1, MPI_INT, MASTER_PROCESS, cartesianComm);
	MPI_Bcast(&grey, 1, MPI_INT, MASTER_PROCESS, cartesianComm);
	MPI_Bcast(&rows, 1, MPI_INT, MASTER_PROCESS, cartesianComm);
	MPI_Bcast(&columns, 1, MPI_INT, MASTER_PROCESS, cartesianComm);
	MPI_Bcast(&best_fit_rows, 1, MPI_INT, MASTER_PROCESS, cartesianComm);

    //1. Find process row and column offsets and vector definition for the input data
    
    int row_index = (comm_rk / (comm_sz / best_fit_rows))* rows;
    int column_index = (comm_rk % (comm_sz / best_fit_rows)) * columns;

    //swapping between source and destination vectors, using a temp vector
    char *source_vec = NULL, *destination_vec = NULL, *temp_vec = NULL;

    //multiplier shall be 3 for RGB input pictures, 1 for GREY input pictures
    unsigned int multiplier = (grey == 1) ? 1 : 3;

    int total_sz = width * height * multiplier;

    source_vec = calloc((rows + 2) * (columns + 2) * multiplier, sizeof(char));
    destination_vec = calloc((rows + 2) * (columns + 2) * multiplier, sizeof(char));

    assert(source_vec != NULL && destination_vec != NULL);

	printf("I am the process %d %d, %d, %d, %d, %s with starting row index = %d and starting column index = %d\n", comm_rk, width, height, loops, grey, picture, row_index, column_index);
    
    //2. Convolution filter definition
    float filter[3][3] = {{1/16.0, 2/16.0, 1/16.0},
                          {2/16.0, 4/16.0, 2/16.0},
                          {1/16.0, 2/16.0, 1/16.0}};


    //3. Parallel read of the input file "image"
    
    MPI_File picture_file = NULL;
    MPI_File_open(cartesianComm, picture, MPI_MODE_RDONLY, MPI_INFO_NULL, &picture_file);

    /*
     * Each process reads the corresponding block of data from the file, and stores it
     * to the appropriate index
     */

    for (i = 0; i < rows; i++) {
        MPI_File_seek(picture_file, multiplier * ((row_index + i) * width + column_index), MPI_SEEK_SET);
        MPI_File_read(picture_file, source_vec + multiplier * columns * i + multiplier, multiplier * columns, MPI_BYTE, &status);
    }

    MPI_File_close(&picture_file);

    printf("read file\n");


    // Create columns for each process
	int columns_number_based_on_type, columns_contiguous, rows_number_based_on_type, blocklength;
	columns_number_based_on_type = (columns + 2) * multiplier;
    columns_contiguous = columns * multiplier;
	rows_number_based_on_type = columns * multiplier;
	blocklength = multiplier;

	MPI_Type_vector(rows, blocklength, columns_number_based_on_type, MPI_BYTE, &send_col);
	MPI_Type_commit(&send_col);
	MPI_Type_contiguous(rows_number_based_on_type, MPI_BYTE, &send_row);
	MPI_Type_commit(&send_row);

    char *outputImage = calloc(strlen("out_image.raw") + 1, sizeof(char));
    strncpy(outputImage, "out_image.raw", strlen("out_image.raw"));

    printf("%s\n", outputImage);
    MPI_File picture_file_out = NULL;
    MPI_File_open(cartesianComm, outputImage, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &picture_file_out);

    /*
     * Each process reads the corresponding block of data from the file, and stores it
     * to the appropriate index
     */

    for (i = 0; i < rows; i++) {
        MPI_File_seek(picture_file_out, multiplier * ((row_index + i) * width + column_index), MPI_SEEK_SET);
        MPI_File_write(picture_file_out, source_vec + multiplier * columns * i + multiplier, multiplier * columns, MPI_BYTE, &status);
    }

    MPI_File_close(&picture_file_out);

   

    free(source_vec);
    free(destination_vec);
    free(outputImage);
    free(picture);

	MPI_Finalize();
	return 0;
}
