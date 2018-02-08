#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <assert.h>
#include "mpi.h"
#include <stdint.h>

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

void convolute(unsigned char *src, unsigned char *dst, int start_row, int start_column, int end_row, int end_column, float h[3][3], int multiplier) {

    int i = 0, j = 0;
    float red = 0.0, green = 0.0, blue = 0.0;

    for (i = start_row; i < end_row - 1; i++) {
        for (j = start_column; j < end_column - 1; j++) {
            if (multiplier == 1) {
                dst[i * end_column + j] = h[0][0] * src[(i - 1) * end_column + j-1] +
                    h[0][1] * src[(i - 1) * end_column + j] +
                    h[0][2] * src[(i - 1) * end_column + j+1] +
                    h[1][0] * src[i * end_column + j-1] +
                    h[1][1] * src[i * end_column + j] +
                    h[1][2] * src[i * end_column + j+1] +
                    h[2][0] * src[(i + 1) * end_column + j-1] +
                    h[2][1] * src[(i + 1) * end_column + j] +
                    h[2][2] * src[(i + 1) * end_column + j+1];
            } else {
                red = h[0][0] * src[(i - 1) * end_column * multiplier + j * multiplier - multiplier] +
                    h[0][1] * src[(i - 1) * end_column * multiplier + j * multiplier] +
                    h[0][2] * src[(i - 1) * end_column * multiplier + j * multiplier + multiplier] +
                    h[1][0] * src[i * end_column * multiplier + j * multiplier - multiplier] +
                    h[1][1] * src[i * end_column * multiplier + j * multiplier] +
                    h[1][2] * src[i * end_column * multiplier + j * multiplier + multiplier] +
                    h[2][0] * src[(i + 1) * end_column * multiplier + j * multiplier - multiplier] +
                    h[2][1] * src[(i + 1) * end_column * multiplier + j * multiplier] +
                    h[2][2] * src[(i + 1) * end_column * multiplier + j * multiplier + multiplier];
                green = h[0][0] * src[(i - 1) * end_column * multiplier + j * multiplier - multiplier + 1] +
                    h[0][1] * src[(i - 1) * end_column * multiplier + j * multiplier + 1] +
                    h[0][2] * src[(i - 1) * end_column * multiplier + j * multiplier + multiplier + 1] +
                    h[1][0] * src[i * end_column * multiplier + j * multiplier - multiplier + 1] +
                    h[1][1] * src[i * end_column * multiplier + j * multiplier + 1] +
                    h[1][2] * src[i * end_column * multiplier + j * multiplier + multiplier + 1] +
                    h[2][0] * src[(i + 1) * end_column * multiplier + j * multiplier - multiplier + 1] +
                    h[2][1] * src[(i + 1) * end_column * multiplier + j * multiplier + 1] +
                    h[2][2] * src[(i + 1) * end_column * multiplier + j * multiplier + multiplier + 1];
                blue = h[0][0] * src[(i - 1) * end_column * multiplier + j * multiplier - multiplier + 2] +
                    h[0][1] * src[(i - 1) * end_column * multiplier + j * multiplier + 2] +
                    h[0][2] * src[(i - 1) * end_column * multiplier + j * multiplier + multiplier + 2] +
                    h[1][0] * src[i * end_column * multiplier + j * multiplier - multiplier + 2] +
                    h[1][1] * src[i * end_column * multiplier + j * multiplier + 2] +
                    h[1][2] * src[i * end_column * multiplier + j * multiplier + multiplier + 2] +
                    h[2][0] * src[(i + 1) * end_column * multiplier + j * multiplier - multiplier + 2] +
                    h[2][1] * src[(i + 1) * end_column * multiplier + j * multiplier + 2] +
                    h[2][2] * src[(i + 1) * end_column * multiplier + j * multiplier + multiplier + 2];
                dst[i * end_column * multiplier + j * multiplier] = red;
                dst[i * end_column * multiplier + j * multiplier + 1] = green;
                dst[i * end_column * multiplier + j * multiplier + 2] = blue;
            }
        }
    }
}

int main(int argc, char **argv) {
    int comm_sz = 0, comm_rk = 0;
    int width = 0, height = 0;
    int loops = 0, grey = 0;
    char *picture = NULL;
    int rows = 0, columns = 0;
    int proc_row = -1, proc_col = -1;
    int i = 0;
    int north, south, east, west;
    unsigned char* se_north_pos, se_south_pos, se_east_pos, se_west_pos, rcv_north_pos, rcv_south_pos, rcv_east_pos, rcv_west_pos;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rk);


    // General Case
    MPI_Request se_north_request;
    MPI_Request se_east_request;
    MPI_Request se_south_request;
    MPI_Request se_west_request;
    MPI_Request rcv_north_request;
    MPI_Request rcv_east_request;
    MPI_Request rcv_south_request;
    MPI_Request rcv_west_request;
    // Boundary Case - Single Elements
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

    north = south = east = west = MPI_PROC_NULL;
    MPI_Dims_create(comm_sz, DIMENSIONALITY, dims);
    MPI_Cart_create(MPI_COMM_WORLD, DIMENSIONALITY, dims, periods, reorder, &cartesianComm);
		MPI_Cart_shift(cartesianComm, 1, 1, &west, &east);
		MPI_Cart_shift(cartesianComm, 0, 1, &north, &south);

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

    printf("%d, %d, \n", row_index, column_index);

    //swapping between source and destination vectors, using a temp vector
    unsigned char *source_vec = NULL, *destination_vec = NULL, *temp_vec = NULL;

    //multiplier shall be 3 for RGB input pictures, 1 for GREY input pictures
    unsigned int multiplier = (grey == 1) ? 1 : 3;

    int total_sz = width * height * multiplier;

    source_vec = calloc((rows + 2) * (columns + 2) * multiplier, sizeof(unsigned char));
    destination_vec = calloc((rows + 2) * (columns + 2) * multiplier, sizeof(unsigned char));

    assert(source_vec != NULL && destination_vec != NULL);

    printf("I am the process %d %d, %d, %d, %d, %s with starting row index = %d and starting column index = %d\n", comm_rk, width, height, loops, grey, picture, row_index, column_index);

    //2. Convolution filter definition
    float filter[3][3] = {{1/16.0, 2/16.0, 1/16.0},
        {2/16.0, 4/16.0, 2/16.0},
        {1/16.0, 2/16.0, 1/16.0}};

//    float filter[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};

    //3. Parallel read of the input file "image"

    MPI_File picture_file = NULL;
    MPI_File_open(cartesianComm, picture, MPI_MODE_RDONLY, MPI_INFO_NULL, &picture_file);

    /*
     * Each process reads the corresponding block of data from the file, and stores it
     * to the appropriate index
     */

    for (i = 1; i <= rows; i++) {
        MPI_File_seek(picture_file, multiplier * ((row_index + i-1) * width + column_index), MPI_SEEK_SET);
        MPI_File_read(picture_file, source_vec + multiplier * (columns + 2) * i + multiplier, multiplier * columns, MPI_BYTE, &status);
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

    for (i = 0; i < loops; i++) {
        // Calculate what each process should send

        unsigned char* se_north_pos = source_vec + (columns + 2) * multiplier + multiplier;
        unsigned char* rcv_north_pos = source_vec + multiplier;
        unsigned char* se_south_pos = source_vec + (rows) * (columns + 2)* multiplier + multiplier;
        unsigned char* rcv_south_pos = source_vec + (rows + 1) * (columns + 2)* multiplier + multiplier;
        unsigned char* se_east_pos = source_vec + (2 * columns + 2) * multiplier;
        unsigned char* rcv_east_pos = source_vec + (2 * columns + 2)* multiplier + multiplier;
        unsigned char* se_west_pos = source_vec + (columns + 2) * multiplier + multiplier;
        unsigned char* rcv_west_pos = source_vec + (columns + 2) * multiplier;
  			//North
  			MPI_Isend(se_north_pos, 1, send_row, north, 0, cartesianComm, &se_north_request);
  			MPI_Irecv(rcv_north_pos, 1, send_row, north, 0, cartesianComm, &rcv_north_request);
  			//South
  			MPI_Isend(se_south_pos, 1, send_row, south, 0, cartesianComm, &se_south_request);
  			MPI_Irecv(rcv_south_pos, 1, send_row, south, 0, cartesianComm, &rcv_south_request);
  			//East
  			MPI_Isend(se_east_pos, 1, send_col, east, 0, cartesianComm, &se_east_request);
  			MPI_Irecv(rcv_east_pos, 1, send_col, east, 0, cartesianComm, &rcv_east_request);
  			//West
  			MPI_Isend(se_west_pos, 1, send_col, west, 0, cartesianComm, &se_west_request);
  			MPI_Irecv(rcv_west_pos, 1, send_col, west, 0, cartesianComm, &rcv_west_request);

        convolute(source_vec, destination_vec, 1, 1, rows + 2, columns + 2, filter, multiplier);

  			if(north != MPI_PROC_NULL) {
  				MPI_Wait(&rcv_north_request, MPI_STATUS_IGNORE);
          convolute(source_vec, destination_vec, 1, 2, 1, columns+1, filter, multiplier);
  			}
  			if(south != MPI_PROC_NULL) {
  				MPI_Wait(&rcv_south_request, MPI_STATUS_IGNORE);
          convolute(source_vec, destination_vec, rows + 2, 2, rows + 2, columns + 1, filter, multiplier);
  			}
  			if(east != MPI_PROC_NULL) {
  				MPI_Wait(&rcv_east_request, MPI_STATUS_IGNORE);
          convolute(source_vec, destination_vec, 2, columns + 2, rows + 1, columns + 2, filter, multiplier);
  			}
  			if(west != MPI_PROC_NULL) {
  				MPI_Wait(&rcv_west_request, MPI_STATUS_IGNORE);
          convolute(source_vec, destination_vec, 2, 1, rows + 1, 1, filter, multiplier);
  			}

        MPI_Wait(&se_north_request, MPI_STATUS_IGNORE);
  			MPI_Wait(&se_south_request, MPI_STATUS_IGNORE);
  			MPI_Wait(&se_east_request, MPI_STATUS_IGNORE);
  			MPI_Wait(&se_west_request, MPI_STATUS_IGNORE);

        temp_vec = source_vec;
        source_vec = destination_vec;
        destination_vec = temp_vec;
    }

    char *outputImage = calloc(strlen("out_image.raw") + 1, sizeof(char));
    strncpy(outputImage, "out_image.raw", strlen("out_image.raw"));

    printf("%s\n", outputImage);
    MPI_File picture_file_out = NULL;
    MPI_File_open(cartesianComm, outputImage, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &picture_file_out);

    /*
     * Each process reads the corresponding block of data from the file, and stores it
     * to the appropriate index
     */

    for (i = 1; i <= rows; i++) {
        MPI_File_seek(picture_file_out, multiplier * ((row_index + i-1) * width + column_index), MPI_SEEK_SET);
        MPI_File_write(picture_file_out, source_vec + multiplier * (columns + 2) * i + multiplier, multiplier * columns, MPI_BYTE, &status);
    }

    MPI_File_close(&picture_file_out);

    free(source_vec);
    free(destination_vec);
    free(outputImage);
    free(picture);

    MPI_Finalize();
}
