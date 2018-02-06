#include <bitset>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

cv::Mat vecToMat(const std::vector< std::vector <uint8_t> > img_vec)
{
  const uint16_t height = img_vec.size();
  const uint16_t width = img_vec[0].size();
  cv::Mat img_cv(height, width, CV_8UC1, cv::Scalar(0));
  for(uint16_t x = 0; x < height; x++)
    for(uint16_t y = 0; y < width; y++)
    {
      img_cv.at<uint8_t>(x, y) = img_vec[x][y];
    }
  return img_cv;
}

int main(int argc, char** argv)
{
  if(argc < 2)
  {
    std::cout << "Missing input argument!" << std::endl;
    return -1;
  }

  std::ifstream img_file(argv[1], std::ios::in | std::ios::binary);

  // Read header, little-endian type
  uint16_t width, height;
  uint8_t pixel_bits;
  img_file.read(reinterpret_cast<char *> (&width), sizeof(width));
  img_file.read(reinterpret_cast<char *> (&height), sizeof(height));
  img_file.read(reinterpret_cast<char *> (&pixel_bits), sizeof(pixel_bits));
  printf("(W, H, BYTE_STEP): (%d, %d, %d)\n", height, width, pixel_bits / 255);

  // Read image pixel values
  std::vector< std::vector<uint8_t> > img_src;
  img_src.resize(height, std::vector<uint8_t>(width, pixel_bits));
  const size_t pixel_size = sizeof(pixel_bits);
  uint8_t pixel_value;
  for(uint16_t x = 0; x < height; x++)
    for(uint16_t y = 0; y < width; y++)
    {
      img_file.read(reinterpret_cast<char *>(&pixel_value), pixel_size);
      img_src[x][y] = pixel_value;
    }

  cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
  cv::imshow("image", vecToMat(img_src));
  cv::waitKey(-1);
  return 0;
}