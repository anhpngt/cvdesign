#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#define HEIGHT 100
#define WIDTH  100

using namespace std;

int main(int argc, char** argv)
{
  std::string param_file = "./eigen_full.yaml";
  if(argc < 2)
    cout << "Using default input file path: " << param_file << endl;
  else
  {
    param_file = argv[1];
    cout << "Input file: " << param_file << endl;
  }

  cv::FileStorage fs(param_file, cv::FileStorage::READ);
  cv::Mat mean(0, 0, CV_64FC1);
  cv::Mat eigen_values(0, 0, CV_64FC1);
  cv::Mat eigen_vectors(0, 0, CV_64FC1);

  std::string out_file = "./eigen.yaml";
  cv::FileStorage fso(out_file, cv::FileStorage::WRITE);

  fs["mean"] >> mean;
  fso << "mean" << mean;
  mean.deallocate();

  fs["eigenvalues"] >> eigen_values;
  fso << "eigenvalues" << eigen_values(cv::Range(0, 200), cv::Range::all());    // [200 x 1]
  eigen_values.deallocate();

  fs["eigenvectors"] >> eigen_vectors;
  eigen_vectors = eigen_vectors(cv::Range(0, 200), cv::Range::all());           // [200 x 10000] from [10000 x 10000]
  cv::transpose(eigen_vectors, eigen_vectors);
  fso << "eigenvectors" << eigen_vectors;  // [10000 x 200]
  eigen_vectors.deallocate();

  cout << "Output: " << out_file << endl;

  fs.release();
  fso.release();

  return 0;
}