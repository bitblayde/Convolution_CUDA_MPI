#include <mpi.h>

#define cimg_use_jpeg
#include "CImg.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <ctime>

using namespace cimg_library;
using namespace std;


vector<vector<float>> gaussian_kernel(){
  vector<vector<float>> kernel(5, std::vector<float>(5, 0));

  float sigma = 3.0;
  double r_value, s_value = 2.0 * sigma * sigma;

  double resultado = 0.0;

  for (int i = -2; i <= 2; i++) {
    for (int j = -2; j <= 2; j++) {
        r_value = sqrt(i * i + j * j);
        kernel[i + 2][j + 2] = (exp(-(r_value * r_value) / s_value)) / (M_PI * s_value);
        resultado += kernel[i + 2][j + 2];
    }
  }


  for (int i = 0; i < 5; i++){
     for (int j = 0; j < 5; j++){
         kernel[i][j] /= resultado;
     }
  }


  return kernel;
}

CImg<float> convolution(const CImg<float> &img){
    float x1, y1, resultado, current_p;
    vector<vector<float>> kernel = gaussian_kernel();
    CImg<float> imagen_resultado(img.width(), img.height(), 1, 1, 1);
    imagen_resultado = imagen_resultado.get_normalize( 0, 1 );

    for(int i = 1; i <= img.width() - 1; i++){
      for(int j = 1; j <= img.height() - 1; j++){

      resultado = 0.0f;
      current_p = 0;

      for(int di = -1; di <= 3; di++){
        for(int dj = -1; dj <= 3; dj++){

          if ( j + dj >= img.height() || i + di >= img.width() ){
            current_p = img( i, j, 0, 0 );
          }
          else{
            current_p = img( i + di , j + dj, 0, 0 );
          }


          resultado += kernel[di + 1][dj + 1] * current_p;
        }
      }
      imagen_resultado(i, j, 0, 0) = resultado;
    }

  }
  return imagen_resultado;
}

int get_pixels(const CImg<float> &src){
  int pixels{0};

  for(int i = 0; i < src.height(); i++)
    for(int j = 0; j < src.width(); j++)
      ++pixels;

  return pixels;
}

CImg<float> join_images(const vector<CImg<float>> &src, const int &processors){
  CImg<float> dst;
  for (int i = 0; i < processors; i++)
    dst.append( src[i], 'y' );

  return dst;
}

vector<CImg<float>> get_sub_images(const CImg<float> &src, const int &processors){
  vector<CImg<float>> dst;

  int h = src.height();
  int w = src.width();
  int n_ini = 0, n_end = src.height() / processors;
  int n = src.height() / processors;

  CImg<float> current_image;

  if(processors > 1){

    for (int i = 0; i < processors; i++){
      current_image = src.get_crop( 0, n_ini, w, n_end );
      dst.emplace_back( current_image );

      n_ini = n_end;
      n_end += src.height() / processors;

      if ( processors - 1 == i ) n_end = h;
    }

  }

  else{

    dst.emplace_back( src );

  }

  return dst;
}



int main(int argc, char** argv){
    int rank, processors;
    CImg<float> image, im_resultado;

    vector<CImg<float>> sub_images;

    if (argc != 2){
      exit(-1);
    }

    string filePath = argv[1];
    image.load(filePath.c_str());

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    auto dst = get_sub_images(image, processors);

    for(int i = rank; i < processors; i++){
      sub_images.push_back( convolution(dst[i]) );
    }

    MPI_Finalize();

    end = std::chrono::system_clock::now();

    if (rank == 0){
      im_resultado = join_images(sub_images, processors );
      im_resultado.save("resultado_parallel.jpg");
      cout << "Píxeles " << get_pixels(im_resultado) << endl;
      int elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>
                                 (end-start).count();

      cout << "tiempo transcurrido sad: " << elapsed_seconds << " ms.\n";
      cout << "Procesos " << processors << endl;
    }
}
