#include <stdio.h> 
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#define BLOCK_SIZE 16
#define CONVERGENCE_CHECK 1

__global__ void convoluteBlock(unsigned char *src, unsigned char *dst, int x, int y, int multiplier) {

    int x_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int y_dim = blockIdx.y * blockDim.y + threadIdx.y;

    float h[3][3] = {{1/16.0, 2/16.0, 1/16.0},
        {2/16.0, 4/16.0, 2/16.0},
        {1/16.0, 2/16.0, 1/16.0}};
    float red = 0.0, green = 0.0, blue = 0.0;
    if (x_dim >= 0 && x_dim < y && y_dim >= 0 && y_dim < x) {
        if (multiplier == 1) {
            dst[x_dim* x + y_dim] = h[0][0] * src[(x_dim- 1) * x + y_dim-1] +
                h[0][1] * src[(x_dim- 1) * x + y_dim] +
                h[0][2] * src[(x_dim- 1) * x + y_dim+1] +
                h[1][0] * src[x_dim* x + y_dim-1] +
                h[1][1] * src[x_dim* x + y_dim] +
                h[1][2] * src[x_dim* x + y_dim+1] +
                h[2][0] * src[(x_dim+ 1) * x + y_dim-1] +
                h[2][1] * src[(x_dim+ 1) * x + y_dim] +
                h[2][2] * src[(x_dim+ 1) * x + y_dim+1];
        } else {
            red = h[0][0] * src[(x_dim - 1) * x * multiplier + y_dim * multiplier - multiplier] +
                h[0][1] * src[(x_dim - 1) * x * multiplier + y_dim * multiplier] +
                h[0][2] * src[(x_dim - 1) * x * multiplier + y_dim * multiplier + multiplier] +
                h[1][0] * src[x_dim * x * multiplier + y_dim * multiplier - multiplier] +
                h[1][1] * src[x_dim * x * multiplier + y_dim * multiplier] +
                h[1][2] * src[x_dim * x * multiplier + y_dim * multiplier + multiplier] +
                h[2][0] * src[(x_dim + 1) * x * multiplier + y_dim * multiplier - multiplier] +
                h[2][1] * src[(x_dim + 1) * x * multiplier + y_dim * multiplier] +
                h[2][2] * src[(x_dim + 1) * x * multiplier + y_dim * multiplier + multiplier];
            green = h[0][0] * src[(x_dim - 1) * x * multiplier + y_dim * multiplier - multiplier + 1] +
                h[0][1] * src[(x_dim - 1) * x * multiplier + y_dim * multiplier + 1] +
                h[0][2] * src[(x_dim - 1) * x * multiplier + y_dim * multiplier + multiplier + 1] +
                h[1][0] * src[x_dim * x * multiplier + y_dim * multiplier - multiplier + 1] +
                h[1][1] * src[x_dim * x * multiplier + y_dim * multiplier + 1] +
                h[1][2] * src[x_dim * x * multiplier + y_dim * multiplier + multiplier + 1] +
                h[2][0] * src[(x_dim + 1) * x * multiplier + y_dim * multiplier - multiplier + 1] +
                h[2][1] * src[(x_dim + 1) * x * multiplier + y_dim * multiplier + 1] +
                h[2][2] * src[(x_dim + 1) * x * multiplier + y_dim * multiplier + multiplier + 1];
            blue = h[0][0] * src[(x_dim - 1) * x * multiplier + y_dim * multiplier - multiplier + 2] +
                h[0][1] * src[(x_dim - 1) * x * multiplier + y_dim * multiplier + 2] +
                h[0][2] * src[(x_dim - 1) * x * multiplier + y_dim * multiplier + multiplier + 2] +
                h[1][0] * src[x_dim * x * multiplier + y_dim * multiplier - multiplier + 2] +
                h[1][1] * src[x_dim * x * multiplier + y_dim * multiplier + 2] +
                h[1][2] * src[x_dim * x * multiplier + y_dim * multiplier + multiplier + 2] +
                h[2][0] * src[(x_dim + 1) * x * multiplier + y_dim * multiplier - multiplier + 2] +
                h[2][1] * src[(x_dim + 1) * x * multiplier + y_dim * multiplier + 2] +
                h[2][2] * src[(x_dim + 1) * x * multiplier + y_dim * multiplier + multiplier + 2];
            dst[x_dim * x * multiplier + y_dim * multiplier] = red;
            dst[x_dim * x * multiplier + y_dim * multiplier + 1] = green;
            dst[x_dim * x * multiplier + y_dim * multiplier + 2] = blue;            
        }
    }
}

__global__ void convergence_grey(unsigned char *src, unsigned char *dst, int x, int y, char *convbool, int multiplier) {
    int x_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int y_dim = blockIdx.y * blockDim.y + threadIdx.y;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    /*Use of shared memory for the convergence check of the current thread's block*/
    __shared__ char blockconvalues[BLOCK_SIZE][BLOCK_SIZE];

    if (0 <= x_dim && x_dim < y && 0 <= y_dim && y_dim < x) {

        if (dst[x_dim * x + y_dim] != src[x_dim * x + y_dim])
            blockconvalues[threadIdx.x][threadIdx.y] = 0;
        else
            blockconvalues[threadIdx.x][threadIdx.y] = 1;
        
        __syncthreads();

        /*First thread of the block, checks if every thread of the block converges*/
        if (threadIdx.x == 0 && threadIdx.y == 0) {

            int blockconv = 1;
            for (int i = 0; i < BLOCK_SIZE; i++) {
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    if (blockconvalues[i][j] != 1) {
                        blockconv = 0;
                        break;
                    }
                }
                if (blockconv == 0)
                    break;
            }

            if (blockconv == 1)
                convbool[blockId] = 1;
            else
                convbool[blockId] = 0;

        }
    }
}

__global__ void convergence_rgb(unsigned char *src, unsigned char *dst, int x, int y, char *convbool, int multiplier) {
    int x_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int y_dim = blockIdx.y * blockDim.y + threadIdx.y;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    /*Use of shared memory for the convergence check of the current thread's block*/
    __shared__ char blockconvalues[BLOCK_SIZE][BLOCK_SIZE * 3];

    if (0 <= x_dim && x_dim < x && 0 <= y_dim && y_dim < y) {

        if (dst[x_dim * x * multiplier + y_dim * multiplier] != src[x_dim * x * multiplier + y_dim * multiplier])
            blockconvalues[threadIdx.x][threadIdx.y] = 0;

        else
            blockconvalues[threadIdx.x][threadIdx.y] = 1; 

        __syncthreads();

        /*First thread of the block, checks if every thread of the block converges*/
        if (threadIdx.x == 0 && threadIdx.y == 0) {

            int blockconv = 1;
            for (int i = 0; i < BLOCK_SIZE; i++) {
                for (int j = 0; j < BLOCK_SIZE * 3; j += 3) {
                    if (blockconvalues[i][j] != 1 || blockconvalues[i][j+1] != 1 || blockconvalues[i][j+2] != 1) {
                        blockconv = 0;
                        break;

                    }

                }
                if (blockconv == 0)
                    break;

            }

            if (blockconv == 1)
                convbool[blockId] = 1;
            else
                convbool[blockId] = 0;


        }
    }
}

extern "C" void convolute(unsigned char *vector, int x, int y, int multiplier, int loops) {
    unsigned char *vector_a, *vector_b, *temp;
    char *convbool, *convboolhost;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialize arrays
    //printf("%d %d %d\n", x, y, x * y);

    convboolhost = (char *)calloc((x * y * multiplier) / BLOCK_SIZE, sizeof(char));
    assert(convboolhost != NULL);

    cudaMalloc(&vector_a, x * y * multiplier * sizeof(unsigned char));
    cudaMalloc(&vector_b, x * y * multiplier * sizeof(unsigned char));
    assert(vector_a != NULL);
    assert(vector_b != NULL);

    cudaMalloc(&convbool, sizeof(char) * ((x * y * multiplier) / BLOCK_SIZE));
    assert(convbool != NULL);

    cudaMemcpy(vector_a, vector, x * y * multiplier * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(vector_b, 0, x * y * multiplier * sizeof(unsigned char));

    int blocksperlinex = (int)ceil((double)((x * multiplier)/ BLOCK_SIZE));
    int blocksperliney = (int)ceil((double)(y / BLOCK_SIZE));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(blocksperliney, blocksperlinex);

    int i = 0;

    int totalconv = 0, first_conv = -1;;
    cudaEventRecord(start);
    for (i = 0; i < loops; i++) {
        if (i > 0) {
            temp = vector_a;
            vector_a = vector_b;
            vector_b = temp;
        }
        convoluteBlock<<<dimGrid, dimBlock>>>(vector_a, vector_b, x, y, multiplier);

        if (i % CONVERGENCE_CHECK == 0 & i > 0) {
            for (int j = 0; j < (x * y * multiplier) / BLOCK_SIZE; j++)
                convboolhost[i] = 0;

            cudaMemcpy(convbool, convboolhost, sizeof(char) * ((x * y * multiplier) / BLOCK_SIZE), cudaMemcpyHostToDevice);
            if (multiplier == 1)
                convergence_grey<<<dimGrid, dimBlock>>>(vector_a, vector_b, x, y, convbool, multiplier);
            else 
                convergence_rgb<<<dimGrid, dimBlock>>>(vector_a, vector_b, x, y, convbool, multiplier);
            cudaMemcpy(convboolhost, convbool, sizeof(char) * ((x * y * multiplier) / BLOCK_SIZE), cudaMemcpyDeviceToHost);

            for (int j = 0; j < (x * y * multiplier) / BLOCK_SIZE; j++) {
                if (convboolhost[i] != 0)
                    totalconv = 1;
                else {
                    totalconv = 0;
                    break;
                }
            }
        }

        if (totalconv == 1 && first_conv == -1) {
            first_conv = i;
        }

        cudaThreadSynchronize();
    }

    cudaEventRecord(stop);
    cudaMemcpy(vector, vector_a, x * y * multiplier * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    if (first_conv >= 0)
        printf("Convergence at %d\n", first_conv);
    else
        printf("No convergence\n");

    cudaFree(vector_a);
    cudaFree(vector_b);
    float msecs = 0.0f;
    cudaEventElapsedTime(&msecs, start, stop);
    printf("Elapsed time = %3.2lf secs\n", msecs / 1000.0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
