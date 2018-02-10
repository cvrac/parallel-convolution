#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <assert.h>
#include "mpi.h"
#include <omp.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define MASTER_PROCESS 0
#define DIMENSIONALITY 2
#define CONVERGENCE_CHECK 5

void convolute(unsigned char *src, unsigned char *dst, int start_row, int start_column, int end_row, int end_column, float h[3][3], int multiplier, int height);

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

inline void convolute(unsigned char *src, unsigned char *dst, int start_row, int start_column, int end_row, int end_column, float h[3][3], int multiplier, int height) {

    int i = 0, j = 0;
    float red = 0.0, green = 0.0, blue = 0.0;

    #pragma omp parallel for shared(src,dst) schedule(static) collapse(3)
    for (i = start_row; i < end_row - 1; i++) {
        for (j = start_column; j < end_column - 1; j++) {
            if (multiplier == 1) {
                dst[i * height + j] = h[0][0] * src[(i - 1) * height + j-1] +
                    h[0][1] * src[(i - 1) * height + j] +
                    h[0][2] * src[(i - 1) * height + j+1] +
                    h[1][0] * src[i * height + j-1] +
                    h[1][1] * src[i * height + j] +
                    h[1][2] * src[i * height + j+1] +
                    h[2][0] * src[(i + 1) * height + j-1] +
                    h[2][1] * src[(i + 1) * height + j] +
                    h[2][2] * src[(i + 1) * height + j+1];
            } else {
                red = h[0][0] * src[(i - 1) * height * multiplier + j * multiplier - multiplier] +
                    h[0][1] * src[(i - 1) * height * multiplier + j * multiplier] +
                    h[0][2] * src[(i - 1) * height * multiplier + j * multiplier + multiplier] +
                    h[1][0] * src[i * height * multiplier + j * multiplier - multiplier] +
                    h[1][1] * src[i * height * multiplier + j * multiplier] +
                    h[1][2] * src[i * height * multiplier + j * multiplier + multiplier] +
                    h[2][0] * src[(i + 1) * height * multiplier + j * multiplier - multiplier] +
                    h[2][1] * src[(i + 1) * height * multiplier + j * multiplier] +
                    h[2][2] * src[(i + 1) * height * multiplier + j * multiplier + multiplier];
                green = h[0][0] * src[(i - 1) * height * multiplier + j * multiplier - multiplier + 1] +
                    h[0][1] * src[(i - 1) * height * multiplier + j * multiplier + 1] +
                    h[0][2] * src[(i - 1) * height * multiplier + j * multiplier + multiplier + 1] +
                    h[1][0] * src[i * height * multiplier + j * multiplier - multiplier + 1] +
                    h[1][1] * src[i * height * multiplier + j * multiplier + 1] +
                    h[1][2] * src[i * height * multiplier + j * multiplier + multiplier + 1] +
                    h[2][0] * src[(i + 1) * height * multiplier + j * multiplier - multiplier + 1] +
                    h[2][1] * src[(i + 1) * height * multiplier + j * multiplier + 1] +
                    h[2][2] * src[(i + 1) * height * multiplier + j * multiplier + multiplier + 1];
                blue = h[0][0] * src[(i - 1) * height * multiplier + j * multiplier - multiplier + 2] +
                    h[0][1] * src[(i - 1) * height * multiplier + j * multiplier + 2] +
                    h[0][2] * src[(i - 1) * height * multiplier + j * multiplier + multiplier + 2] +
                    h[1][0] * src[i * height * multiplier + j * multiplier - multiplier + 2] +
                    h[1][1] * src[i * height * multiplier + j * multiplier + 2] +
                    h[1][2] * src[i * height * multiplier + j * multiplier + multiplier + 2] +
                    h[2][0] * src[(i + 1) * height * multiplier + j * multiplier - multiplier + 2] +
                    h[2][1] * src[(i + 1) * height * multiplier + j * multiplier + 2] +
                    h[2][2] * src[(i + 1) * height * multiplier + j * multiplier + multiplier + 2];
                dst[i * height * multiplier + j * multiplier] = red;
                dst[i * height * multiplier + j * multiplier + 1] = green;
                dst[i * height * multiplier + j * multiplier + 2] = blue;
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
    int i = 0;
    int north, south, east, west;
    int north_west, north_east, south_west, south_east;
    int first_converge = 1;
    srand(time(NULL));

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
    MPI_Request se_nw_request;
    MPI_Request se_ne_request;
    MPI_Request se_se_request;
    MPI_Request se_sw_request;
    MPI_Request rcv_nw_request;
    MPI_Request rcv_ne_request;
    MPI_Request rcv_se_request;
    MPI_Request rcv_sw_request;
    MPI_Status status;

    MPI_Datatype send_col, send_row, send_corner;

    // Cartesian topology definition
    MPI_Comm cartesianComm;
    int dims[DIMENSIONALITY] = {0, 0};
    int periods[2] = {0, 0};
    int reorder = 1;

    north = south = east = west = MPI_PROC_NULL;
    MPI_Dims_create(comm_sz, DIMENSIONALITY, dims);
    MPI_Cart_create(MPI_COMM_WORLD, DIMENSIONALITY, dims, periods, reorder, &cartesianComm);
    MPI_Comm_set_errhandler(cartesianComm, MPI_ERRORS_RETURN);
    MPI_Cart_shift(cartesianComm, 1, 1, &west, &east);
    MPI_Cart_shift(cartesianComm, 0, 1, &north, &south);
    int coords[DIMENSIONALITY];
    MPI_Cart_coords(cartesianComm, comm_rk, DIMENSIONALITY, coords);

    //North West
    int nw_coords[DIMENSIONALITY];
    nw_coords[0] = coords[0] - 1; nw_coords[1] = coords[1] - 1;
    if(MPI_Cart_rank(cartesianComm, nw_coords, &north_west))
        north_west = MPI_PROC_NULL;
    //North East
    int ne_coords[DIMENSIONALITY];
    ne_coords[0] = coords[0] - 1; ne_coords[1] = coords[1] + 1;
    if(MPI_Cart_rank(cartesianComm, ne_coords, &north_east))
        north_east = MPI_PROC_NULL;
    // South West
    int sw_coords[DIMENSIONALITY];
    sw_coords[0] = coords[0] + 1; sw_coords[1] = coords[1] - 1;
    if(MPI_Cart_rank(cartesianComm, sw_coords, &south_west))
        south_west = MPI_PROC_NULL;

    // South East
    int se_coords[DIMENSIONALITY];
    se_coords[0] = coords[0] + 1; se_coords[1] = coords[1] + 1;
    if(MPI_Cart_rank(cartesianComm, se_coords, &south_east))
        south_east = MPI_PROC_NULL;
    // Variables setup
    int best_fit_rows;
    if (comm_rk == MASTER_PROCESS) {
        width = atoi(argv[1]); height = atoi(argv[2]);
        loops = atoi(argv[3]);
        if (!strcmp(argv[4], "grey"))
            grey = 1;

        best_fit_rows = best_fit(height, width, comm_sz);
        if (!best_fit_rows) {
            MPI_Abort(cartesianComm, 1);
            return 1;
        }
        rows = height / best_fit_rows;
        columns = width / (comm_sz / best_fit_rows);
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
    unsigned char *source_vec = NULL, *destination_vec = NULL, *temp_vec = NULL;

    //multiplier shall be 3 for RGB input pictures, 1 for GREY input pictures
    unsigned int multiplier = (grey == 1) ? 1 : 3;

    source_vec = calloc((rows + 2) * (columns + 2) * multiplier, sizeof(unsigned char));
    destination_vec = calloc((rows + 2) * (columns + 2) * multiplier, sizeof(unsigned char));

    assert(source_vec != NULL && destination_vec != NULL);

    //2. Convolution filter definition
    float filter[3][3] = {{1/16.0, 2/16.0, 1/16.0},
        {2/16.0, 4/16.0, 2/16.0},
        {1/16.0, 2/16.0, 1/16.0}};

    //    float filter[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};

    //3. Parallel read of the input file "image"

    if (!strcmp("random", picture)) {
      if (grey) {
        for (i = 1; i <= rows; i++) {
          for (int j = 1 ; j <= columns; j++) {
            source_vec[i * (columns + 2) * multiplier + j * multiplier] = rand() % 254;
          }
        }
      } else {
          for (i = 1; i <= rows; i++) {
            for (int j = 1 ; j <= columns; j++) {
              source_vec[i * (columns + 2) * 3 + j * 3] = rand() % 254;
              source_vec[i * (columns + 2) * 3 + j * 3 + 1] = rand() % 254;
              source_vec[i * (columns + 2) * 3 + j * 3 + 2] = rand() % 254;
            }
          }
      }

    } else {
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
    }


    // Create columns for each process
    int columns_number_based_on_type, rows_number_based_on_type, blocklength;
    unsigned int convergence_at_loop = 0;
    columns_number_based_on_type = (columns + 2) * multiplier;
    rows_number_based_on_type = columns * multiplier;
    blocklength = multiplier;

    MPI_Type_vector(rows, blocklength, columns_number_based_on_type, MPI_BYTE, &send_col);
    MPI_Type_commit(&send_col);
    MPI_Type_contiguous(rows_number_based_on_type, MPI_BYTE, &send_row);
    MPI_Type_commit(&send_row);

    MPI_Type_vector(1, blocklength, 1, MPI_BYTE, &send_corner);
    MPI_Type_commit(&send_corner);

    MPI_Barrier(cartesianComm);

    double start_time = MPI_Wtime();

    int t = 0, total_convergence = 0;
    for (t = 0; t < loops; t++) {
        // Calculate what each process should send
        // Datatypes calculation
        unsigned char* se_north_pos  = source_vec + (columns + 2) * multiplier + multiplier;
        unsigned char* rcv_north_pos = source_vec + multiplier;
        unsigned char* se_south_pos  = source_vec + (rows) * (columns + 2) * multiplier + multiplier;
        unsigned char* rcv_south_pos = source_vec + (rows + 1) * (columns + 2)* multiplier + multiplier;
        unsigned char* se_east_pos   = source_vec + (2 * columns + 2) * multiplier;
        unsigned char* rcv_east_pos  = source_vec + (2 * columns + 2)* multiplier + multiplier;
        unsigned char* se_west_pos   = source_vec + (columns + 2) * multiplier + multiplier;
        unsigned char* rcv_west_pos  = source_vec + (columns + 2) * multiplier;

        // Corner Elements
        unsigned char* rcv_nw_pos = source_vec;
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

        // Corner Elements
        //NorthWest
        MPI_Isend(se_north_pos, 1, send_corner, north_west, 0, cartesianComm, &se_nw_request);
        MPI_Irecv(rcv_nw_pos, 1, send_corner, north_west, 0, cartesianComm, &rcv_nw_request);
        //NorthEast
        MPI_Isend(se_east_pos, 1, send_corner, north_east, 0, cartesianComm, &se_ne_request);
        MPI_Irecv((unsigned char*)(source_vec + (columns + 1) * multiplier), 1, send_corner, north_east, 0, cartesianComm, &rcv_ne_request);
        //SouthWest
        MPI_Isend((unsigned char*)(source_vec + (rows * (columns + 2) + 1) * multiplier), 1, send_corner, south_west, 0, cartesianComm, &se_sw_request);
        MPI_Irecv((unsigned char*)(source_vec + ((rows + 1) * (columns + 2) * multiplier)), 1, send_corner, south_west, 0, cartesianComm, &rcv_sw_request);
        //SouthEast
        MPI_Isend((unsigned char*)(source_vec + ((rows * (columns + 2) + columns)* multiplier)), 1, send_corner, south_east, 0, cartesianComm, &se_se_request);
        MPI_Irecv((unsigned char*)(source_vec + (rows  + 1) * (columns + 2) * multiplier), 1, send_corner, south_east, 0, cartesianComm, &rcv_se_request);
        //Inner Convolute
        convolute(source_vec, destination_vec, 2, 2, rows + 1, columns + 1, filter, multiplier, columns + 2);

        // Outer Elements Convolute (Wait for Receive)
        MPI_Wait(&rcv_north_request, MPI_STATUS_IGNORE);
        convolute(source_vec, destination_vec, 1, 2, 3, columns + 1, filter, multiplier, columns + 2); // North
        MPI_Wait(&rcv_south_request, MPI_STATUS_IGNORE);
        convolute(source_vec, destination_vec, rows, 2, rows + 3, columns + 1, filter, multiplier, columns + 2); // South
        MPI_Wait(&rcv_east_request, MPI_STATUS_IGNORE);
        convolute(source_vec, destination_vec, 2, columns, rows + 1, columns + 2, filter, multiplier, columns + 2); // East
        MPI_Wait(&rcv_west_request, MPI_STATUS_IGNORE);
        convolute(source_vec, destination_vec, 2, 1, rows + 1, 3, filter, multiplier, columns + 2); // West
        // Wait for corner items
        MPI_Wait(&rcv_nw_request, MPI_STATUS_IGNORE);
        convolute(source_vec, destination_vec, 1, 1, 3, 3, filter, multiplier, columns + 2); // NorthWest
        MPI_Wait(&rcv_ne_request, MPI_STATUS_IGNORE);
        convolute(source_vec, destination_vec, 1, columns, 3, columns + 2, filter, multiplier, columns + 2); // NorthEast
        MPI_Wait(&rcv_se_request, MPI_STATUS_IGNORE);
        convolute(source_vec, destination_vec, rows, columns, rows + 2, columns + 2, filter, multiplier, columns + 2); // SouthEast
        MPI_Wait(&rcv_sw_request, MPI_STATUS_IGNORE);
        convolute(source_vec, destination_vec, rows, 1, rows + 2, 3, filter, multiplier, columns + 2); // SouthWest


        /*Convergence check, using AllReduce*/
        int local_convergence = 1;
        if (t % CONVERGENCE_CHECK == 0) {
            int i, j;
            if (grey) {
              for (i = 0; i < rows + 2; i++)
                  for (j = 0; j < columns + 2; j++)
                      if (destination_vec[i * (columns + 2) + j ] != source_vec[i * (columns + 2) + j])
                          local_convergence = 0;
            } else {
              //RGB
                for (i = 0; i < rows + 2; i++)
                    for (j = 0; j < columns + 2; j++)
                        if (destination_vec[i * (columns + 2) * 3 + j * 3] != source_vec[i * (columns + 2) * 3 + j * 3] ||
                            destination_vec[i * (columns + 2) * 3 + j * 3 + 1] != source_vec[i * (columns + 2) * 3 + j * 3 + 1] ||
                            destination_vec[i * (columns + 2) * 3 + j * 3 + 2] != source_vec[i * (columns + 2) * 3 + j * 3 + 2]) {
                              local_convergence = 0;
                        }
            }

            MPI_Allreduce(&local_convergence, &total_convergence, 1, MPI_INT, MPI_LAND, cartesianComm);
            if (total_convergence > 0 && first_converge) {
              convergence_at_loop = t;
              first_converge = 0;
            }
        }

        // Wait for Send
        MPI_Wait(&se_north_request, MPI_STATUS_IGNORE);
        MPI_Wait(&se_south_request, MPI_STATUS_IGNORE);
        MPI_Wait(&se_east_request, MPI_STATUS_IGNORE);
        MPI_Wait(&se_west_request, MPI_STATUS_IGNORE);
        MPI_Wait(&se_nw_request, MPI_STATUS_IGNORE);
        MPI_Wait(&se_ne_request, MPI_STATUS_IGNORE);
        MPI_Wait(&se_sw_request, MPI_STATUS_IGNORE);
        MPI_Wait(&se_se_request, MPI_STATUS_IGNORE);

        temp_vec = source_vec;
        source_vec = destination_vec;
        destination_vec = temp_vec;

    }

    MPI_Barrier(cartesianComm);
    double end_time = MPI_Wtime();

    char *outputImage = calloc(strlen("out_image.raw") + 1, sizeof(char));
    strncpy(outputImage, "out_image.raw", strlen("out_image.raw"));

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

    if (comm_rk == MASTER_PROCESS) {
      printf("Elapsed time = %3.2lf seconds", end_time - start_time);
      if (!convergence_at_loop) {
        printf(" (no convergence)\n");
      } else {
        printf(" with convergence at loop %d\n", convergence_at_loop);
      }
      printf("Blurred image file: %s\n", outputImage);
    }
    free(source_vec);
    free(destination_vec);
    free(outputImage);
    free(picture);
    MPI_Finalize();
    return 0;
}