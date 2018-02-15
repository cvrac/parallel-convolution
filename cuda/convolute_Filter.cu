#include <stdio.h> 
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#define BLOCK_SIZE 16

#define FRACTION_CEILING(numerator, denominator) ((numerator+denominator-1)/denominator)

__global__ void convoluteBlock(unsigned char *src, unsigned char *dst, int x, int y, int multiplier) {

    /* x = width, y = x*/
    int x_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int y_dim = blockIdx.y * blockDim.y + threadIdx.y;

    float h[3][3] = {{1/16.0, 2/16.0, 1/16.0},
        {2/16.0, 4/16.0, 2/16.0},
        {1/16.0, 2/16.0, 1/16.0}};
    float red = 0.0, green = 0.0, blue = 0.0;
    if (x_dim > 0 && x_dim < y - 1 && y_dim > 0 && y_dim < x - 1) {
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

extern "C" void convolute(unsigned char *vector, int x, int y, int multiplier, int loops) {
    unsigned char *vector_a, *vector_b, *temp;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialize arrays
    printf("%d %d %d\n", x, y, x * y);

    cudaMalloc(&vector_a, x * y * multiplier * sizeof(unsigned char));
    cudaMalloc(&vector_b, x * y * multiplier * sizeof(unsigned char));
    assert(vector_a != NULL);
    assert(vector_b != NULL);


    cudaMemcpy(vector_a, vector, x * y * multiplier * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(vector_b, 0, x * y * multiplier * sizeof(unsigned char));
    printf("%d %d\n", FRACTION_CEILING(x * multiplier, BLOCK_SIZE), FRACTION_CEILING(y, BLOCK_SIZE));

//    int blocksperlinex = (int)ceil((double)(x * multiplier / BLOCK_SIZE));
//    int blocksperliney = (int)ceil((double)(y / BLOCK_SIZE));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(FRACTION_CEILING(y, BLOCK_SIZE), FRACTION_CEILING(x * multiplier, BLOCK_SIZE));
    //dim3 dimGrid(blocksperliney, blocksperlinex);

    printf("%d\n", multiplier);
    int i = 0;

    printf("%d\n", loops);
    for (i = 0; i < loops; i++) {
        if (i > 0) {
            temp = vector_a;
            vector_a = vector_b;
            vector_b = temp;
        }
        convoluteBlock<<<dimGrid, dimBlock>>>(vector_a, vector_b, x, y, multiplier);

    }
    cudaThreadSynchronize();

    cudaEventRecord(stop);
    cudaMemcpy(vector, vector_a, x * y * multiplier * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(vector_a);
    cudaFree(vector_b);
}
