#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <assert.h>
#include "mpi.h"

#define MASTER_PROCESS 0
#define DIMENSIONALITY 2

inline unsigned int best_fit(int rows, int columns, int processes);
 
int main(int argc, char **argv) {


	int comm_sz = 0, comm_rk = 0;	
	int width = 0, height = 0;
	int loops = 0, grey = 0;
	char *picture = NULL;

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
	
	MPI_Datatype send_col, send_row;

	// Cartesian topology definition

	MPI_Comm cartesianComm;
	int dims[DIMENSIONALITY] = {0, 0};
	int periods[2] = {0, 0};
	int reorder = 1;
	MPI_Dims_create(comm_sz, DIMENSIONALITY, dims);
	MPI_Cart_create(MPI_COMM_WORLD, DIMENSIONALITY, dims, periods, reorder, &cartesianComm); 


	// Variables setup

	if (comm_rk == MASTER_PROCESS) {
		width = atoi(argv[1]); height = atoi(argv[2]);
		loops = atoi(argv[3]);

		if (!strcmp(argv[4], "grey"))
			grey = 1; 

		printf("I am the master process\n"); 
	
	}


	picture = malloc(strlen(argv[5]) * sizeof(char));
	assert( picture != NULL);
		
	strncpy(picture, argv[5], strlen(argv[5]));


	MPI_Bcast(&width, 1, MPI_INT, MASTER_PROCESS, cartesianComm);
	MPI_Bcast(&height, 1, MPI_INT, MASTER_PROCESS, cartesianComm);
	MPI_Bcast(&loops, 1, MPI_INT, MASTER_PROCESS, cartesianComm);
	MPI_Bcast(&grey, 1, MPI_INT, MASTER_PROCESS, cartesianComm);

	

	printf("I am the process %d %d, %d, %d, %d, %s\n", comm_rk, width, height, loops, grey, picture);

	MPI_Finalize();
	return 0;
}




inline unsigned int best_fit(int rows, int columns, int processes) {

	unsigned int i, rows_it, columns_it;

	for (rows_it = 1; rows_it != workers; rows_it++) {
		
	}
	
}

