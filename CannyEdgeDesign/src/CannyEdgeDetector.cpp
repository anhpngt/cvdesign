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
      else if (input[x][y] < 0) 
      {
        // printf("pixel (%d, %d) of value %.2lf", x, y, input[x][y]);
        output[x][y] = 0;
      }
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
  Vec2d img_blur_, img_Ix_, img_Iy_, img_I_, img_nms_;
  Vec2d gauss_mask_;
  double sigma_;

private:
  void conv2d(const Vec2d, const Vec2d, Vec2d&);
  double computeGaussianWeight(int x, int y, double sigma);
  Vec2d computeGaussKernel(int k_size, double sigma);
  // Vec2d mask_dx_ = {{-1, -2, -1},
  //                   { 0,  0,  0},
  //                   { 1,  2,  1}};
  // Vec2d mask_dy_ = {{-1,  0,  1},
  //                   {-2,  0,  2},
  //                   {-1,  0,  1}}; // easier to visualize
  Vec2d mask_dx_ = {{-0.125, -0.25, -0.125},
                    { 0,         0,      0},
                    { 0.125,  0.25,  0.125}};
  Vec2d mask_dy_ = {{-0.125,     0,  0.125},
                    {-0.25,      0,   0.25},
                    {-0.125,     0,  0.125}};
  double const pi = 3.14159265359;
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
  if(mask_size < 3) 
    mask_size = 3;
  else if(mask_size % 2 == 0) 
    mask_size += 1;

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
  
  // // Gradient smoothing with moving average of 3 pixels
  // for(uint16_t x = 1, x_end = img_I_.size() - 1; x < x_end; x++) 
  //   for(uint16_t y = 1, y_end = img_I_[0].size() - 1; y < y_end; y++)
  //   {
  //     // Compute the gradient direction
  //     double grad_angle = atan2(img_Iy_[x][y], img_Ix_[x][y]) * 180 / pi; // (-180, 180], note that x->down, y->right
      
  //     // Process
  //     int code = abs(int(round(grad_angle / 45.0))) % 4;
  //     switch(code)
  //     {
  //       case 0: // up <-> down
  //         img_I_[x][y] = (img_I_[x-1][y] + img_I_[x][y] + img_I_[x+1][y]) / 3.0;
  //         break;
  //       case 1: // up-left <-> down-right
  //         img_I_[x][y] = (img_I_[x+1][y+1] + img_I_[x][y] + img_I_[x-1][y-1]) / 3.0;
  //         break;
  //       case 2: // left <-> right
  //         img_I_[x][y] = (img_I_[x][y-1] + img_I_[x][y] + img_I_[x][y+1]) / 3.0;
  //         break;
  //       case 3: // up-right <-> down-left
  //         img_I_[x][y] = (img_I_[x-1][y+1] + img_I_[x][y] + img_I_[x+1][y-1]) / 3.0;
  //         break;
  //       default: 
  //         std::cout << "ERROR: grad_angle_code = " << code << std::endl;
  //         exit(-1);
  //     }
  //   }

  // Non-Maximum Suppression
  // note that this will reduce dimensions of img by (2, 2)
  img_nms_.resize(img_I_.size() - 2, std::vector<double>(img_I_[0].size() - 2, 0));
  for(uint16_t x = 1, x_end = img_I_.size() - 1; x < x_end; x++) 
    for(uint16_t y = 1, y_end = img_I_[0].size() - 1; y < y_end; y++)
    {
      // Compute the gradient direction
      double grad_angle = atan2(img_Iy_[x][y], img_Ix_[x][y]) * 180 / pi; // (-180, 180], note that x->down, y->right
      
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

  ////////////////////////////////////////
  //           Post-processing          //
  ////////////////////////////////////////

  // Normalize pixel intensity of img_I_ and img_nms_ for visualization
  double brightest_pixel_val = 0;
  for(uint16_t x = 0, x_end = img_I_.size(); x < x_end; x++) 
    for(uint16_t y = 0, y_end = img_I_[0].size(); y < y_end; y++)
    {
      if(img_I_[x][y] > brightest_pixel_val)
        brightest_pixel_val = img_I_[x][y];
    }
  double mul_val = 255.0 / brightest_pixel_val;
  std::cout << "Multiplying img_I_ by " << mul_val << " to visualize" << std::endl;
  for(uint16_t x = 0, x_end = img_I_.size(); x < x_end; x++) 
    for(uint16_t y = 0, y_end = img_I_[0].size(); y < y_end; y++)
    {
      img_I_[x][y] = img_I_[x][y] * mul_val;
    }
  
  brightest_pixel_val = 0;
  for(uint16_t x = 0, x_end = img_nms_.size(); x < x_end; x++) 
    for(uint16_t y = 0, y_end = img_nms_[0].size(); y < y_end; y++)
    {
      if(img_nms_[x][y] > brightest_pixel_val)
        brightest_pixel_val = img_nms_[x][y];
    }
  mul_val = 255.0 / brightest_pixel_val;
  std::cout << "Multiplying img_nms_ by " << mul_val << " to visualize" << std::endl;
  for(uint16_t x = 0, x_end = img_nms_.size(); x < x_end; x++) 
    for(uint16_t y = 0, y_end = img_nms_[0].size(); y < y_end; y++)
    {
      img_nms_[x][y] = img_nms_[x][y] * mul_val;
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
  cv::namedWindow("Canny Edge Detector", cv::WINDOW_AUTOSIZE);

  cv::Mat cv_src = VecToMat(img_src);
  // cv::Mat cv_blur = VecToMat(imgproc.img_Ix_);
  // cv::Mat cv_I = VecToMat(imgproc.img_Iy_);
  // cv::Mat cv_nms = VecToMat(imgproc.img_I_);
  cv::Mat cv_blur = VecToMat(imgproc.img_blur_);
  cv::Mat cv_I = VecToMat(imgproc.img_I_);
  cv::Mat cv_nms = VecToMat(imgproc.img_nms_);

  // Padding zeros to image edges to concatnate
  int blur_padding = (cv_src.cols - cv_blur.cols) / 2;
  int i_padding = (cv_src.cols - cv_I.cols) / 2;
  int nms_padding = (cv_src.cols - cv_nms.cols) / 2;
  cv::copyMakeBorder(cv_blur, cv_blur, blur_padding, blur_padding, blur_padding, blur_padding, cv::BORDER_CONSTANT, cv::Scalar::all(0));
  cv::copyMakeBorder(cv_I, cv_I, i_padding, i_padding, i_padding, i_padding, cv::BORDER_CONSTANT, cv::Scalar::all(0));
  cv::copyMakeBorder(cv_nms, cv_nms, nms_padding, nms_padding, nms_padding, nms_padding, cv::BORDER_CONSTANT, cv::Scalar::all(0));

  cv::Mat visual;
  std::vector<cv::Mat> output_mat;
    output_mat.push_back(cv_src);
    output_mat.push_back(cv_blur);
    output_mat.push_back(cv_I);
    output_mat.push_back(cv_nms);
  cv::hconcat(output_mat, visual);
  cv::imshow("Canny Edge Detector", visual);
  cv::imwrite("processed_image.jpg", visual);

  cv::waitKey(-1);
  return 0;
}