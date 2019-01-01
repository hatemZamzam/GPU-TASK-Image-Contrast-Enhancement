#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H

// CUDA Runtime
#include <cuda_runtime.h>
// Utility and system includes
#include <helper_cuda.h>
// helper for shared that are common to CUDA Samples
#include <helper_functions.h>
#include <helper_timer.h>

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    

typedef struct{
    int w;
    int h;
    unsigned char * img_r;
    unsigned char * img_g;
    unsigned char * img_b;
} PPM_IMG;

typedef struct{
    int w;
    int h;
    unsigned char * img_y;
    unsigned char * img_u;
    unsigned char * img_v;
} YUV_IMG;


typedef struct
{
    int width;
    int height;
    float * h;
    float * s;
    unsigned char * l;
} HSL_IMG;

    

PPM_IMG read_ppm(const char * path);
void write_ppm(PPM_IMG img, const char * path);
void free_ppm(PPM_IMG img);

// gpu hsl
HSL_IMG gpu_rgb2hsl(PPM_IMG img_in);
PPM_IMG gpu_hsl2rgb(HSL_IMG img_in);
__global__ void gpu_hsl2rgbCal(float * d_h , float * d_s ,unsigned char * d_l , unsigned char * r, unsigned char * g, unsigned char * b, int size, int numOfThreads);
__global__ void gpu_rgb2hslCal(float * d_h , float * d_s ,unsigned char * d_l , unsigned char * d_r, unsigned char * d_g, unsigned char * d_b, int size, int numOfThreads);


__global__ void gpu_yuv2rgbCal(unsigned char * y , unsigned char * u ,unsigned char * v , unsigned char * r, unsigned char * g, unsigned char * b, int size, int numOfThreads);
__global__ void gpu_rbg2yuvCal(unsigned char * d_y , unsigned char * d_u ,unsigned char * d_v , unsigned char * d_r, unsigned char * d_g, unsigned char * d_b, int size, int numOfThreads);

// gpu yuv
YUV_IMG gpu_rgb2yuv(PPM_IMG img_in);
PPM_IMG gpu_yuv2rgb(YUV_IMG img_in);    

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin);

__global__ void gpu_histogram(int * hist_out, unsigned char * img_in, int nbr_bin, int numOfThreads, int img_size);
__global__ void gpu_histogram_equalization(unsigned char * img_out, unsigned char * img_in,
                            int * hist_in, int img_size, int nbr_bin, int numOfThreads, int * lut);


//Contrast enhancement for color images for GPU
PPM_IMG gpu_contrast_enhancement_c_yuv(PPM_IMG img_in);
PPM_IMG gpu_contrast_enhancement_c_hsl(PPM_IMG img_in);


#endif
