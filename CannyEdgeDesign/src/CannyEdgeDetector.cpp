#include <bitset>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

typedef uint8_t pixel_t;
typedef std::vector< std::vector<pixel_t> > Vec2i;
typedef std::vector< std::vector<double> > Vec2d;

///////////////////////////////////////////////////////////////////////////////
Vec2d Vec2iToVec2d(const Vec2i input)
{
  Vec2d output;
  output.resize(input.size(), std::vector<double>(input[0].size(), 0));
  for(uint16_t x = 0; x < output.size(); x++)
    for(uint16_t y = 0; y < output[0].size(); y++)
      output[x][y] = (double)input[x][y];
  return output;
}

Vec2i Vec2dToVec2i(const Vec2d input)
{
  Vec2i output;
  output.resize(input.size(), std::vector<pixel_t>(input[0].size(), 0));
  for(uint16_t x = 0; x < output.size(); x++)
    for(uint16_t y = 0; y < output[0].size(); y++)
    {
      if(input[x][y] > 255) output[x][y] = 255;
      else if (input[x][y] < 0) output[x][y] = 0;
      else output[x][y] = (pixel_t)input[x][y];
    }
  return output;
}

///////////////////////////////////////////////////////////////////////////////
cv::Mat VecToMat(const Vec2i img_vec)
{
  const uint16_t height = img_vec.size();
  const uint16_t width = img_vec[0].size();
  cv::Mat img_cv(height, width, CV_8UC1, cv::Scalar(0));
  for(uint16_t x = 0; x < height; x++)
    for(uint16_t y = 0; y < width; y++)
    {
      img_cv.at<pixel_t>(x, y) = img_vec[x][y];
    }
  return img_cv;
}

cv::Mat VecToMat(const Vec2d img_vec)
{
  return VecToMat(Vec2dToVec2i(img_vec));
}

///////////////////////////////////////////////////////////////////////////////
double computeGaussianWeight(int x, int y, double sigma)
{
  return exp(-(double)(x * x + y * y) / (2.0 * sigma * sigma));
}

///////////////////////////////////////////////////////////////////////////////
class ImageProcessing
{
public:
  ImageProcessing(Vec2i, double);

  Vec2i img_src_;
  Vec2d img_blur_, img_Ix_, img_Iy_, img_I_, img_nms_, img_hys_;
  Vec2d gauss_mask_;
  double sigma_;

private:
  void conv2d(const Vec2d, const Vec2d, Vec2d&);
  Vec2d mask_dx_ = {{-1, -2, -1},
                   { 0,  0,  0},
                   { 1,  2,  1}};
  Vec2d mask_dy_ = {{-1, 0, 1},
                   {-2, 0, 2},
                   {-1, 0, 1}};
  double const pi = 3.14159265359;
};

ImageProcessing::ImageProcessing(Vec2i img, double sigma):
  img_src_(img),
  sigma_(sigma)
{
  ////////////////////////////////////////
  //         Gaussian Smoothing         //
  ////////////////////////////////////////

  // Comput mask size
  if(sigma <= 0)
  {
    std::cout << "ERROR: invalid sigma value!" << std::endl;
    exit(-1);
  }
  int mask_size = ceil(5 * sigma_);
  if(mask_size < 3) mask_size = 3;
  else if(mask_size % 2 == 0) mask_size += 1;
  int mask_center = (mask_size - 1) / 2;

  // Compute mask value
  gauss_mask_.resize(mask_size, std::vector<double>(mask_size, 0));
  double weight_sum = 0;
  for(int x = 0; x < mask_size; x++)
    for(int y = 0; y < mask_size; y++)
    {
      gauss_mask_[x][y] = computeGaussianWeight(x - mask_center, y - mask_center, sigma_);
      weight_sum += gauss_mask_[x][y];
    }
  
  // Normalize mask
  for(int x = 0; x < mask_size; x++)
    for(int y = 0; y < mask_size; y++)
      gauss_mask_[x][y] /= weight_sum;
    
  // Conv2d
  conv2d(Vec2iToVec2d(img_src_), gauss_mask_, img_blur_);
  
  ////////////////////////////////////////
  //        Compute Derivatives         //
  ////////////////////////////////////////

  conv2d(img_blur_, mask_dx_, img_Ix_);  // dx
  conv2d(img_blur_, mask_dy_, img_Iy_);  // dy
  img_I_.resize(img_Ix_.size(), std::vector<double>(img_Ix_[0].size(), 0));
  for(uint16_t x = 0; x < img_I_.size(); x++)
    for(uint16_t y = 0; y < img_I_[0].size(); y++)
    {
      double Ix = img_Ix_[x][y];
      double Iy = img_Iy_[x][y];
      img_I_[x][y] = sqrt((Ix * Ix + Iy * Iy) / 2); // sqrt(dx^2 + dy^2)
    }

  ////////////////////////////////////////
  //       Non-Maximum Suppression      //
  ////////////////////////////////////////
  
  // note that this will reduce dimensions of img by (2, 2)
  img_nms_.resize(img_I_.size() - 2, std::vector<double>(img_I_[0].size() - 2, 0));
  for(uint16_t x = 1, x_end = img_I_.size() - 1; x < x_end; x++) 
    for(uint16_t y = 1, y_end = img_I_[0].size() - 1; y < y_end; y++)
    {
      // Compute the gradient direction
      double grad_angle = atan2(img_Iy_[x][y], img_Ix_[x][y]) * 180 / pi; // (-180, 180], note that x->down, y->right
      
      // TODO: Add smoothing before suppressing
      // Process
      int code = abs(int(round(grad_angle / 45.0))) % 4;
      switch(code)
      {
        case 0: // up <-> down
          if(img_I_[x][y] > img_I_[x-1][y] && img_I_[x][y] > img_I_[x+1][y])
            img_nms_[x-1][y-1] = img_I_[x][y];
          break;
        case 1: // up-left <-> down-right
          if(img_I_[x][y] > img_I_[x-1][y-1] && img_I_[x][y] > img_I_[x+1][y+1])
            img_nms_[x-1][y-1] = img_I_[x][y];
          break;
        case 2: // left <-> right
          if(img_I_[x][y] > img_I_[x][y-1] && img_I_[x][y] > img_I_[x][y+1])
            img_nms_[x-1][y-1] = img_I_[x][y];
          break;
        case 3: // up-right <-> down-left
          if(img_I_[x][y] > img_I_[x-1][y+1] && img_I_[x][y] > img_I_[x+1][y-1])
            img_nms_[x-1][y-1] = img_I_[x][y];
          break;
        default: 
          std::cout << "ERROR: grad_angle_code = " << code << std::endl;
          exit(-1);
      }
    }
}

void ImageProcessing::conv2d(const Vec2d input, const Vec2d mask, Vec2d &output)
{
  if(input.empty())
  {
    std::cout << "Empty input matrix in 2D convolution!" << std::endl;
    exit(-1);
  }
  if(mask.empty())
  {
    std::cout << "Empty mask matrix in 2D convolution!" << std::endl;
    exit(-1);
  }

  // Get output dimensions
  uint16_t out_height = input.size() - mask.size() + 1;
  uint16_t out_width = input[0].size() - mask[0].size() + 1;
  output.resize(out_height, std::vector<double>(out_width, 0));
  printf("Doing 2d convolution (%d, %d) x (%d, %d) >> (%d, %d)\n", 
         int(input[0].size()), int(input.size()), int(mask[0].size()), int(mask.size()), out_width, out_height);

  // Conv
  for(uint16_t x = 0; x < out_height; x++)
    for(uint16_t y = 0; y < out_width; y++)
      for(uint16_t mask_x = 0; mask_x < mask.size(); mask_x++)
        for(uint16_t mask_y = 0; mask_y < mask[0].size(); mask_y++)
          output[x][y] += input[x + mask_x][y + mask_y] * mask[mask_x][mask_y];
  return;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  if(argc < 3)
  {
    std::cout << "Missing input arguments!" << std::endl;
    std::cout << "./cannyEdgeDetector \"image.raw\" \"pixel_standard_dev\"" << std::endl;
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
  Vec2i img_src;
  img_src.resize(height, std::vector<pixel_t>(width, 0));
  const size_t pixel_size = sizeof(pixel_bits);
  pixel_t pixel_value;
  for(uint16_t x = 0; x < height; x++)
    for(uint16_t y = 0; y < width; y++)
    {
      img_file.read(reinterpret_cast<char *>(&pixel_value), pixel_size);
      img_src[x][y] = pixel_value;
    }

  // Call processing
  ImageProcessing imgproc(img_src, std::stod(argv[2]));

  printf("Standard deviation: %lf\n", imgproc.sigma_);
  std::cout << "Guassian smoothing using mask: " << std::endl;
  for(uint16_t i = 0; i < imgproc.gauss_mask_.size(); i++)
  {
    for(uint16_t j = 0; j < imgproc.gauss_mask_.size(); j++)
      printf("  %.8lf", imgproc.gauss_mask_[i][j]);
    std::cout << std::endl;
  }

  // Visualize using OpenCV
  printf("Visualizing...\n");
  cv::namedWindow("Source", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Gaussian Smoothed", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Derivative", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Thin Edge", cv::WINDOW_AUTOSIZE);
  cv::Mat cv_src = VecToMat(img_src);
  cv::Mat cv_blur = VecToMat(imgproc.img_blur_);
  cv::Mat cv_I = VecToMat(imgproc.img_I_);
  cv::Mat cv_nms = VecToMat(imgproc.img_nms_);
  cv::imshow("Source", cv_src);
  cv::imshow("Gaussian Smoothed", cv_blur);
  cv::imshow("Derivative", cv_I);
  cv::imshow("Thin Edge", cv_nms);

  cv::waitKey(-1);
  return 0;
}