#include <iostream>

#define cimg_use_jpeg
#include "CImg.h"
#include <vector>

#include <chrono>
#include <ctime>

using namespace cimg_library;
using namespace std;

vector<vector<float>> gaussian_kernel(){
  vector<vector<float>> kernel(5, std::vector<float>(5, 0));

  float sigma = 1.0;
  double r, s = 2.0 * sigma * sigma;


  double sum = 0.0;

  for (int x = -2; x <= 2; x++) {
    for (int y = -2; y <= 2; y++) {
        r = sqrt(x * x + y * y);
        kernel[x + 2][y + 2] = (exp(-(r * r) / s)) / (M_PI * s);
        sum += kernel[x + 2][y + 2];
    }
  }


  for (int i = 0; i < 5; ++i)
     for (int j = 0; j < 5; ++j)
         kernel[i][j] /= sum;


  return kernel;
}

int get_threshold(int thresh, int current_i){
    if (current_i < 0){
        return current_i+thresh;
    }

    if(current_i >= thresh){
        return current_i-thresh;
    }

   return current_i;
}

CImg<float> convolution(const CImg<float> &img, vector<vector<float>> kernel){
  float x1, y1, resultado, current_p;
  CImg<float> dst(img.width(), img.height(), 1, 1, 1);

    for(int i = 1; i <= img.width() - 1; i++){
      for(int j = 1; j <= img.height() - 1; j++){

      resultado = 0.0f;
      current_p = 0;

      for(int ki = -1; ki <= 3; ki++){
        for(int kj = -1; kj <= 3; kj++){
          if ( i + ki >= img.width() || j + kj >= img.height() ){
            current_p = img( i, j, 0, 0 );
          }
          else{
            current_p = img( i + ki , j + kj, 0, 0 );
          }


          resultado += kernel[ki + 1][kj + 1] * current_p;
        }
      }
      dst(i, j, 0, 0) = resultado;
    }

  }
  return dst;
}




int main(int argc, char** argv){
  if (argc != 2){
    exit(-1);
  }

  CImg<float> image, dst;
  std::chrono::time_point<std::chrono::system_clock> start, end;

  string filePath = argv[1];
  image.load(filePath.c_str());

  // Image preprocessing
  image = image.get_normalize( 0, 1 );

  // Kernel
  vector<vector<float>> kernel = gaussian_kernel();

  // Convolución
  start = std::chrono::system_clock::now();
  dst = convolution(image, gaussian_kernel());
  end = std::chrono::system_clock::now();

  int elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>
                             (end-start).count();

  cout << "tiempo transcurrido: " << elapsed_seconds << " ms.\n";

  dst.save("secuencial_convolution.bmp");

  CImgDisplay main_disp(dst, "The image");

  while (!main_disp.is_closed()){
    main_disp.wait();
  }

  return 0;
}
