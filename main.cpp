#include <stdio.h>
#include <string.h>
#include <stdlib.h>
// CUDA Runtime
#include <cuda_runtime.h>
// Utility and system includes
#include <helper_cuda.h>
// helper for shared that are common to CUDA Samples
#include <helper_functions.h>
#include <helper_timer.h>

#include "hist-equ.h"

void run_gpu_color_test(PPM_IMG img_in);

int main(){
    
    // for initialization
    int * init;
    cudaMalloc(&init, 0);
    
    PPM_IMG img_ibuf_c;
    
    printf("====================================================\n");
    printf("    Running contrast enhancement for color images     \n");
    printf("====================================================\n");
    img_ibuf_c = read_ppm("in.ppm");
    run_gpu_color_test(img_ibuf_c);
    free_ppm(img_ibuf_c);
    
    return 0;
}

void run_gpu_color_test(PPM_IMG img_in)
{
    printf("-------------------------------------\n");
    StopWatchInterface *timer=NULL;
    PPM_IMG img_obuf_hsl, img_obuf_yuv, img_obuf_rgb;
    
    printf("Starting GPU processing\n");
    printf("-------------------------------------\n");
    
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    img_obuf_hsl = gpu_contrast_enhancement_c_hsl(img_in);
    sdkStopTimer(&timer);
    printf("HSL processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
    
    write_ppm(img_obuf_hsl, "gpu_out_hsl.ppm");
    
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    img_obuf_yuv = gpu_contrast_enhancement_c_yuv(img_in);
    sdkStopTimer(&timer);
    printf("YUV processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
    
    write_ppm(img_obuf_yuv, "gpu_out_yuv.ppm");
    
    free_ppm(img_obuf_hsl);
    free_ppm(img_obuf_yuv);
}


PPM_IMG read_ppm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    char *ibuf;
    PPM_IMG result;
    int v_max, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    /*Skip the magic number*/
    fscanf(in_file, "%s", sbuf);


    //result = malloc(sizeof(PPM_IMG));
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         = (char *)malloc(3 * result.w * result.h * sizeof(char));

    
    fread(ibuf,sizeof(unsigned char), 3 * result.w*result.h, in_file);

    for(i = 0; i < result.w*result.h; i ++){
        result.img_r[i] = ibuf[3*i + 0];
        result.img_g[i] = ibuf[3*i + 1];
        result.img_b[i] = ibuf[3*i + 2];
    }
    
    fclose(in_file);
    free(ibuf);
    
    return result;
}

void write_ppm(PPM_IMG img, const char * path){
    FILE * out_file;
    int i;
    
    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_r[i];
        obuf[3*i + 1] = img.img_g[i];
        obuf[3*i + 2] = img.img_b[i];
    }
    out_file = fopen(path, "wb");
    fprintf(out_file, "P6\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(obuf,sizeof(unsigned char), 3*img.w*img.h, out_file);
    fclose(out_file);
    free(obuf);
}

void free_ppm(PPM_IMG img)
{
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}

