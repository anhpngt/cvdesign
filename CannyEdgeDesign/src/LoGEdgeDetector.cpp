#include <algorithm>
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
class ImageProcessing
{
public:
  ImageProcessing(Vec2i, double);

  Vec2i img_src_;
  Vec2d img_blur_, img_LoG_, img_edge_;
  Vec2d gauss_mask_, log_mask_;
  double sigma_;

private:
  void conv2d(const Vec2d, const Vec2d, Vec2d&);
  double computeGaussianWeight(int x, int y, double sigma);
  double computeLoGWeight(int x, int y, double sigma);
  Vec2d computeGaussKernel(int k_size, double sigma);
  Vec2d computeLoGKernel(int size, double sigma);
  // Vec2d mask_dx_ = {{-1, -2, -1},
  //                   { 0,  0,  0},
  //                   { 1,  2,  1}};
  // Vec2d mask_dy_ = {{-1,  0,  1},
  //                   {-2,  0,  2},
  //                   {-1,  0,  1}};
  Vec2d mask_dx_ = {{-0.125, -0.25, -0.125},
                    { 0,         0,      0},
                    { 0.125,  0.25,  0.125}};
  Vec2d mask_dy_ = {{-0.125,     0,  0.125},
                    {-0.25,      0,   0.25},
                    {-0.125,     0,  0.125}};
  double const pi = 3.14159265359;
  double edge_strength_threshold = 0.0;
  bool use_kernel_for_log = true;
};

ImageProcessing::ImageProcessing(Vec2i img, double sigma):
  img_src_(img),
  sigma_(sigma)
{
  ////////////////////////////////////////
  //         Gaussian Smoothing         //
  ////////////////////////////////////////

  // Compute mask size
  if(sigma <= 0)
  {
    std::cout << "ERROR: invalid sigma value!" << std::endl;
    exit(-1);
  }
  int mask_size = ceil(5 * sigma_);
  if(mask_size < 3) mask_size = 3;
  else if(mask_size % 2 == 0) mask_size += 1;

  // Compute mask value
  gauss_mask_ = computeGaussKernel(mask_size, sigma_);  

  std::cout << "Standard deviation: " << sigma_ << std::endl;
  std::cout << "Guassian smoothing using mask: " << std::endl;
  for(uint16_t i = 0; i < gauss_mask_.size(); i++)
  {
    for(uint16_t j = 0; j < gauss_mask_.size(); j++)
      printf("  %11.8lf", gauss_mask_[i][j]);
    std::cout << std::endl;
  }
  std::cout << std::endl;

  // Conv2d
  conv2d(Vec2iToVec2d(img_src_), gauss_mask_, img_blur_);
  
  ////////////////////////////////////////
  //          Compute Laplacian         //
  ////////////////////////////////////////

  // Laplacian is computed using Sobel operator twice in each direction
  // Laplacian can also be computed using a mask
  if(!use_kernel_for_log)
  {
    Vec2d img_Ix, img_Ix2, img_Iy, img_Iy2;
    conv2d(img_blur_, mask_dx_, img_Ix);
    conv2d(img_Ix, mask_dx_, img_Ix2);
    conv2d(img_blur_, mask_dy_, img_Iy);
    conv2d(img_Iy, mask_dy_, img_Iy2);

    img_LoG_.resize(img_Ix2.size(), std::vector<double>(img_Ix2[0].size(), 0));
    for(uint16_t x = 0; x < img_LoG_.size(); x++)
      for(uint16_t y = 0; y < img_LoG_[0].size(); y++)
        img_LoG_[x][y] = img_Ix2[x][y] + img_Iy2[x][y]; // d(dx) + d(dy)
  }
  else
  {
    log_mask_ = computeLoGKernel(mask_size, sigma_);
    std::cout << "Laplacian of Gaussian mask: " << std::endl;
    for(uint16_t i = 0; i < log_mask_.size(); i++)
    {
      for(uint16_t j = 0; j < log_mask_.size(); j++)
        printf("  %11.8lf", log_mask_[i][j]);
      std::cout << std::endl;
    }
    std::cout << std::endl;

    conv2d(img_blur_, log_mask_, img_LoG_);
  }

  ////////////////////////////////////////
  //         Find Zero-crossing         //
  ////////////////////////////////////////

  img_edge_.resize(img_LoG_.size() - 2, std::vector<double>(img_LoG_[0].size() - 2, 0));
  for(uint16_t x = 1, x_end = img_edge_.size() - 1; x < x_end; x++)
    for(uint16_t y = 1, y_end = img_edge_[0].size() - 1; y < y_end; y++)
    {
      // Zero-crossing: 0 value or having opposite sign neighbors
      if(img_LoG_[x][y] == 0)
      {
        // Assign the biggest gradient of opposite neighbor couple (if they have different signs)
        double grad_br_tl = img_LoG_[x+1][y+1]*img_LoG_[x-1][y-1] < 0 ? std::fabs(img_LoG_[x+1][y+1] - img_LoG_[x-1][y-1]) : 0;
        double grad_bb_tt = img_LoG_[x+1][y]*img_LoG_[x-1][y] < 0 ? std::fabs(img_LoG_[x+1][y] - img_LoG_[x-1][y]) : 0;
        double grad_bl_tr = img_LoG_[x+1][y-1]*img_LoG_[x-1][y+1] < 0 ? std::fabs(img_LoG_[x+1][y-1] - img_LoG_[x-1][y+1]) : 0;
        double grad_rr_ll = img_LoG_[x][y+1]*img_LoG_[x][y-1] < 0 ? std::fabs(img_LoG_[x][y+1] - img_LoG_[x][y-1]) : 0;
        double grad_array[] = {grad_br_tl, grad_bb_tt, grad_bl_tr, grad_rr_ll};
        img_edge_[x][y] = *std::max_element(grad_array, grad_array + 4);
      }
      else // look for opposite sign neighbor
      {
        double neighbor_array[] = {img_LoG_[x-1][y-1], img_LoG_[x-1][y], img_LoG_[x-1][y+1],
                                   img_LoG_[x][y-1],                     img_LoG_[x][y+1],
                                   img_LoG_[x+1][y-1], img_LoG_[x+1][y], img_LoG_[x+1][y+1]};
        if(img_LoG_[x][y] > 0) // then, find most neg = min
        {
          double most_neg_neighbor = *std::min_element(neighbor_array, neighbor_array + 8);
          if(most_neg_neighbor < 0)
            img_edge_[x][y] = img_LoG_[x][y] - most_neg_neighbor;
          else
            img_edge_[x][y] = 0;
        }
        else // then, find most pos = max
        {
          double most_pos_neighbor = *std::max_element(neighbor_array, neighbor_array + 8);
          if(most_pos_neighbor > 0)
            img_edge_[x][y] = -img_LoG_[x][y] + most_pos_neighbor;
          else
            img_edge_[x][y] = 0;
        }
      }

      // Filter edge if its strength is below threshold
      if(img_edge_[x][y] < edge_strength_threshold)
        img_edge_[x][y] = 0;
    }
  
  ////////////////////////////////////////
  //           Post-processing          //
  ////////////////////////////////////////

  // Normalize (0-255 range) pixel intensity of img_LoG_ and img_edge_ for visualization
  double high_val = -999.0;
  double low_val = 999.0;
  for(uint16_t x = 0, x_end = img_LoG_.size(); x < x_end; x++) 
    for(uint16_t y = 0, y_end = img_LoG_[0].size(); y < y_end; y++)
    {
      if(img_LoG_[x][y] > high_val)
        high_val = img_LoG_[x][y];
      if(img_LoG_[x][y] < low_val)
        low_val = img_LoG_[x][y];
    }
  
  double mul_val = 255.0 / (high_val - low_val);
  std::cout << "Offsetting then multiplying img_LoG_ by " << -low_val << " and " << mul_val << " to visualize." << std::endl;
  for(uint16_t x = 1, x_end = img_LoG_.size() - 1; x < x_end; x++) 
    for(uint16_t y = 1, y_end = img_LoG_[0].size() - 1; y < y_end; y++)
    {
      img_LoG_[x][y] = img_LoG_[x][y] - low_val;
      img_LoG_[x][y] = img_LoG_[x][y] * mul_val;
    }
  
  high_val = 0;
  for(uint16_t x = 0, x_end = img_edge_.size(); x < x_end; x++) 
    for(uint16_t y = 0, y_end = img_edge_[0].size(); y < y_end; y++)
    {
      if(img_edge_[x][y] > high_val)
        high_val = img_edge_[x][y];
    }
  mul_val = 255.0 / high_val;
  std::cout << "Multiplying img_edge_ by " << mul_val << " to visualize" << std::endl;
  for(uint16_t x = 0, x_end = img_edge_.size(); x < x_end; x++) 
    for(uint16_t y = 0, y_end = img_edge_[0].size(); y < y_end; y++)
    {
      img_edge_[x][y] = img_edge_[x][y] * mul_val;
    }
}

void ImageProcessing::conv2d(const Vec2d input, const Vec2d mask, Vec2d &output)
{
  if(input.empty() || mask.empty())
  {
    std::cout << "Empty input matrix in 2D convolution!" << std::endl;
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
double ImageProcessing::computeGaussianWeight(int x, int y, double sigma)
{
  return exp(-(double)(x * x + y * y) / (2.0 * sigma * sigma));
}

Vec2d ImageProcessing::computeGaussKernel(int k_size, double sigma)
{
  // Check input params
  if(sigma <= 0)
  {
    std::cout << "Invalid sigma value in Gaussian kernel creation: " << sigma << std::endl;
    exit(-1);
  }

  if(k_size % 2 == 0 || k_size < 3)
  {
    std::cout << "Invalid k_size value in Gaussian kernel creation: " << k_size << std::endl;
    exit(-1);
  }

  Vec2d gauss_kernel;
  gauss_kernel.resize(k_size, std::vector<double>(k_size, 0));
  double weight_sum = 0;
  int mask_center = (k_size - 1) / 2;
  for(int x = 0; x < k_size; x++)
    for(int y = 0; y < k_size; y++)
    {
      gauss_kernel[x][y] = computeGaussianWeight(x - mask_center, y - mask_center, sigma);
      weight_sum += gauss_kernel[x][y];
    }
  
  // Normalize mask
  for(int x = 0; x < k_size; x++)
    for(int y = 0; y < k_size; y++)
      gauss_kernel[x][y] /= weight_sum;
  
  return gauss_kernel;
}

///////////////////////////////////////////////////////////////////////////////
double ImageProcessing::computeLoGWeight(int x, int y, double sigma)
{
  double rho = -(x * x + y * y) / (2.0 * sigma * sigma);
  return -(1 + rho) * exp(rho) / (3.14159265359 * sigma * sigma * sigma * sigma);
}

Vec2d ImageProcessing::computeLoGKernel(int k_size, double sigma)
{
  // Check input params
  if(sigma <= 0)
  {
    std::cout << "Invalid sigma value in LoG kernel creation: " << sigma << std::endl;
    exit(-1);
  }

  if(k_size % 2 == 0 || k_size < 3)
  {
    std::cout << "Invalid k_size value in LoG kernel creation: " << k_size << std::endl;
    exit(-1);
  }

  // Compute kernel
  Vec2d log_kernel;
  log_kernel.resize(k_size, std::vector<double>(k_size, 0));
  double weight_sum = 0;
  int kernel_center = (k_size - 1) / 2;
  for(int x = 0; x < k_size; x++)
    for(int y = 0; y < k_size; y++)
    {
      log_kernel[x][y] = computeLoGWeight(x - kernel_center, y - kernel_center, sigma);
      weight_sum += log_kernel[x][y];
    }
  
  // Zero-mean kernel
  double mean = weight_sum / (k_size * k_size);
  weight_sum = 0;
  for(int x = 0; x < k_size; x++)
    for(int y = 0; y < k_size; y++)
    {
      log_kernel[x][y] -= mean;
      weight_sum += log_kernel[x][y];
    }

  return log_kernel;
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

  // Visualize using OpenCV
  printf("Finished! Visualizing using OpenCV...\n");
  cv::namedWindow("LoG Edge Detector", cv::WINDOW_AUTOSIZE);

  cv::Mat cv_src = VecToMat(img_src);
  cv::Mat cv_blur = VecToMat(imgproc.img_blur_);
  cv::Mat cv_LoG = VecToMat(imgproc.img_LoG_);
  cv::Mat cv_edge = VecToMat(imgproc.img_edge_);

  // Padding zeros to image edges to concatnate
  int blur_padding = (cv_src.cols - cv_blur.cols) / 2;
  int log_padding = (cv_src.cols - cv_LoG.cols) / 2;
  int edge_padding = (cv_src.cols - cv_edge.cols) / 2;
  cv::copyMakeBorder(cv_blur, cv_blur, blur_padding, blur_padding, blur_padding, blur_padding, cv::BORDER_CONSTANT, cv::Scalar::all(0));
  cv::copyMakeBorder(cv_LoG, cv_LoG, log_padding, log_padding, log_padding, log_padding, cv::BORDER_CONSTANT, cv::Scalar::all(0));
  cv::copyMakeBorder(cv_edge, cv_edge, edge_padding, edge_padding, edge_padding, edge_padding, cv::BORDER_CONSTANT, cv::Scalar::all(0));

  cv::Mat visual;
  std::vector<cv::Mat> output_mat;
    output_mat.push_back(cv_src);
    output_mat.push_back(cv_blur);
    output_mat.push_back(cv_LoG);
    output_mat.push_back(cv_edge);
  cv::hconcat(output_mat, visual);
  cv::imshow("LoG Edge Detector", visual);

  cv::waitKey(-1);
  return 0;
}