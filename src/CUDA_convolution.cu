#include <cuda.h>
#include <iostream>
#include <chrono>
#include <ctime>

#include "convolucion_CUDA.cuh"


__global__
void convolution( int w, int h, int *src, int *dst){ //const CImg<float> &img){
    int id_thread = threadIdx.x + blockDim.x * blockIdx.x;

    unsigned int resultado, current_p;

    if (id_thread < w - 2){

      int kernel[5][5] = {
     {1, 4, 7, 4, 1},
     {4, 16, 26, 16, 4},
     {7, 26, 41, 26, 7},
     {4, 16, 26, 16, 4},
     {1, 4, 7, 4, 1}
     };


      for(int k = 2; k <= h - 2; k++){

        resultado = 0.0f;

        for(int i = -2; i <= 2; i++){
          for(int j = -2; j <= 2; j++){

            if( (k + j >= h) || ( i + id_thread >= w ) ){
              current_p = *(src + id_thread * h + k);
            }
            else{
              current_p = *( src + (id_thread + i) * h + (k + j) );
            }

            resultado = resultado + current_p * kernel[i+2][j+2];
          }
        }
        *(dst + id_thread * h + k) = resultado/273;
      }

    }

}

double convolution_interface( int w, int h, int *src){
  clock_t begin_computo, end_computo;

  dim3 block_number((w/32) + 1, 1, 1);
  dim3 block_dim(32, 1, 1);
  int *__convolution_pointer = nullptr;
  int size = sizeof(int)*(w*h);

  int *dst = nullptr;

  cudaMalloc((void **) &__convolution_pointer, size);
  cudaMalloc((void **) &dst, size);

  begin_computo = clock();

  cudaMemcpy(__convolution_pointer, src, size, cudaMemcpyHostToDevice);
  convolution<<< block_number, block_dim >>>(w, h, __convolution_pointer, dst);
  cudaDeviceSynchronize();
  cudaMemcpy(src, dst, size, cudaMemcpyDeviceToHost);

  end_computo = clock();

  cudaFree(__convolution_pointer);
  cudaFree(dst);

  return double(end_computo - begin_computo) / CLOCKS_PER_SEC;
}
