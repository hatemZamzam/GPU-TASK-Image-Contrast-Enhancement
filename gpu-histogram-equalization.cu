#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"


__global__ void gpu_histogram(int * hist_out, unsigned char * img_in, int nbr_bin, int numOfThreads, int img_size){
    
    int i = threadIdx.x + blockDim.x*blockIdx.x;

    for (int k = 0; k < nbr_bin; k ++){
        hist_out[k] = 0;
    }
    
    int start;
    int end;
    //int img_size = (int)(img_in.height*img_in.width);
    //hist_in[x%256] = i;
    /* Get the result image */
    if(i >= img_size) {
       return;
    }
    start = ((img_size/numOfThreads) * i);
    if(numOfThreads == 1) {
       end = (img_size/numOfThreads);
    }
    else {
       end = ((img_size/numOfThreads) * (i+1));
    }

    for(int j = start; j < end; j ++){
        //hist_out[j] = 0;
        //int myBin = img_in[j] % nbr_bin;
        atomicAdd(&(hist_out[img_in[j]]), 1);
    }

}

__global__ void gpu_histogram_equalization(unsigned char * img_out, unsigned char * img_in,
                            int * hist_in, int img_size, int nbr_bin, int numOfThreads, int * lut){


    int i = 0;
    int x = threadIdx.x + blockDim.x*blockIdx.x;

    int start;
    int end;
    //hist_in[x%256] = x;
    /* Get the result image */
    if(x >= img_size) {
       return;
    }
    start = ((img_size/numOfThreads) * x);
    if(numOfThreads == 1) {
       end = (img_size/numOfThreads);
    }
    else {
       end = ((img_size/numOfThreads) * (x+1));
    }
    for(i = start; i < end; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }
}


