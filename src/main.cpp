//#define cimg_use_jpeg
#include "CImg.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <ctime>
#include<tuple>
#include "CUDA_convolution.cuh"

using namespace cimg_library;
using namespace std;

ostream& operator<<(ostream &res, const vector<vector<float>> & kernel){
  for( auto ki : kernel ){
    for( auto kj : ki ){
      res << kj << ", ";
    }
    res << endl;
  }
  return res;
}



std::tuple<CImg<int>, double> get_image(const CImg<int>&src){
  CImg<int> resultado(src.width(), src.height(), 1, 1, 1);

  int size = sizeof(int)*src.height()*src.width();

  int *image_1D = nullptr;
  image_1D = (int *)malloc(size);
  for(auto row = 0; row < src.width(); row++){
    for(auto col = 0; col < src.height(); col++){
      *( image_1D + row * src.height() + col ) = src(row, col, 0, 0);
    }
  }

  auto tiempo = convolution_interface(src.width(), src.height(), image_1D);


  for(auto row = 0; row < src.width(); row++){
    for(auto col = 0; col < src.height(); col++){
      resultado(row, col, 0, 0) = *( image_1D + row * src.height() + col );
    }
  }

  free(image_1D);

  return {resultado, tiempo};
}


int main(int argc, char** argv){
    CImg<int> image, im_resultado;
    std::chrono::time_point<std::chrono::system_clock> start, end;

    if (argc != 3){
      cerr << "Formato de argumentos:" << endl;
      cerr << "<ruta de la imagen> <nombre de la imagen resultante>" << endl;
      exit(-1);
    }

    string filePath = argv[1];
    image.load(filePath.c_str());
    auto [resultado, tiempo] = get_image(image);
    std:: cout << "Tiempo " << tiempo << " s." << endl;
    resultado.save(argv[2]);

  }
