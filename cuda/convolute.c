#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char **argv) {

    unsigned char *vector = NULL;

    char *input_picture = NULL, *output_picture = NULL;
    int width = -1, height = -1, loops = -1, multiplier = 1, random = -1;

    if (argc < 6) {
        printf("Not enough arguments\n");
        return 1;
    } else if (argc == 7) random = 1;

    width = atoi(argv[1]);
    height = atoi(argv[2]);
    loops = atoi(argv[3]);
    if (!strcmp(argv[4], "grey")) multiplier = 1;
    else multiplier = 3;
    input_picture = calloc(strlen(argv[5]) + 1, sizeof(char));
    assert(input_picture != NULL);
    strncpy(input_picture, argv[5], strlen(argv[5]));


    int desc = -1;

    if ((desc = open(input_picture, O_RDONLY)) < 0) {
        fprintf(stderr, "cannot open input picture file\n");
        return 1;
    }

    vector = calloc(width * height * multiplier, sizeof(unsigned char));
    assert(vector != NULL);


    if (random < 0) {
        int read_b = 0;
        for (long i = 0; i < width * height * multiplier; i+= read_b) {
            if ((read_b = read(desc, vector + i, width * height * multiplier - i)) < 0) {
                fprintf(stderr, "Error on reading\n");
                return 1;
            }
        }
    }

    close(desc);

    // Convolution calculation
    
//    convolute();

    output_picture = calloc(strlen("out_image.raw") + 1, sizeof(char));
    assert(output_picture != NULL);
    strncpy(output_picture, "out_image.raw", strlen("out_image.raw"));

    if ((desc = open(output_picture, O_CREAT | O_WRONLY, 0644)) < 0) {
        fprintf(stderr, "Cannot create output picture\n");
        return 1;
    }

    int write_b = 0;
    for (long i = 0; i < width * height * multiplier; i += write_b) {
        if ((write_b = write(desc, vector + i, width * height * multiplier - i)) < 0) {
            fprintf(stderr, "Error on writing\n");
            return 1;
        }
    }

    close(desc);
    free(output_picture); free(input_picture); free(vector);
    output_picture = NULL; input_picture = NULL; vector = NULL;

    return 0;
}
